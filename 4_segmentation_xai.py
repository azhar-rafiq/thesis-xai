"""
RSNA Intracranial Hemorrhage -- XAI Map Generation (GradCAM + U-Net)
Run via sbatch: sbatch 4_segmentation_xai.sh
"""

import gc
import os
import sys
import json
import datetime
import types
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
from training_logger import save_models, log_run

load_dotenv()
SEED = 20260605
tf.random.set_seed(SEED)
np.random.seed(SEED)

CACHE_DIR        = Path(os.getenv('KAGGLE_CACHE')) / 'preprocessed'
MODEL_EXPORT_DIR = '/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models'
IMG_SIZE         = 256
N_CLASSES        = 5
label_cols       = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-4
N_SEG_MAPS = 2000  # positive test samples to save maps for


# ---- load preprocessed data ----
sidecars = sorted(CACHE_DIR.glob(f'*_{IMG_SIZE}_meta.json'))
if not sidecars:
    raise FileNotFoundError(f"No preprocessed cache in {CACHE_DIR}. Run 0_preprocess_rsna.py first.")
meta_path = sidecars[-1]
meta      = json.loads(meta_path.read_text())
base      = str(meta_path).replace('_meta.json', '')

print(f"Loading cache: {meta_path.name}", flush=True)
print(f"  train: {meta['n_train_slices']} slices, {meta['N_TRAIN_PATIENTS']} patients", flush=True)
print(f"  val:   {meta['n_val_slices']} slices,  {meta['N_VAL_PATIENTS']} patients", flush=True)
print(f"  test:  {meta['n_test_slices']} slices,  {meta['N_TEST_PATIENTS']} patients", flush=True)

x_train = np.load(f'{base}_x_train.npy', mmap_mode='r')
y_train = np.load(f'{base}_y_train.npy', mmap_mode='r')
x_val   = np.load(f'{base}_x_val.npy',   mmap_mode='r')
y_val   = np.load(f'{base}_y_val.npy',   mmap_mode='r')
x_test  = np.load(f'{base}_x_test.npy',  mmap_mode='r')
y_test  = np.load(f'{base}_y_test.npy',  mmap_mode='r')

print(f"\nx_train: {x_train.shape}, y_train: {y_train.shape}", flush=True)
print(f"x_val:   {x_val.shape},   y_val:   {y_val.shape}", flush=True)
print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}", flush=True)


# ---- GPU setup ----
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}", flush=True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    details = tf.config.experimental.get_device_details(gpu)
    print(f"  {gpu.name} - {details.get('device_name', 'unknown')} "
          f"(compute {details.get('compute_capability', ('?','?'))[0]}."
          f"{details.get('compute_capability', ('?','?'))[1]})", flush=True)
if not gpus:
    print("WARNING: No GPU detected!", flush=True)


# ---- load CNN checkpoint from step 1 ----
CNN_CKPT = os.path.join(MODEL_EXPORT_DIR, 'rsna_cnn_20260605_05000_auc.keras')
VIT_CKPT = os.path.join(MODEL_EXPORT_DIR, 'rsna_vit_20260608-ZU_08740_auc.keras')
HYBRID_CKPT = os.path.join(MODEL_EXPORT_DIR, 'rsna_hybrid_20260608-AR_09512_auc.keras')

if not os.path.exists(CNN_CKPT):
    raise FileNotFoundError(f"CNN checkpoint not found: {CNN_CKPT}")
print(f"\nLoading CNN checkpoint: {os.path.basename(CNN_CKPT)}", flush=True)
cnn_base = tf.keras.models.load_model(CNN_CKPT)
print(f"  CNN layers: {len(cnn_base.layers)}", flush=True)


def build_unet_from_cnn(cnn_base, freeze_encoder=True):
    """
    grafts a U-Net decoder onto the frozen CNN encoder from step 1.
    skip connections tap relu outputs before each MaxPool (256x, 128x, 64x res).
    GAP over the final seg_maps gives the classification score -- no pixel labels needed.
    returns (classifier_model, segmenter_model) sharing all weights.
    """
    cnn_base.trainable = not freeze_encoder

    # CNN is a Sequential model; .input/.output attributes don't exist until the model
    # is called as a functional layer. trace through layers explicitly instead.
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
    x   = inp
    act_outputs  = []
    drop_outputs = []
    for layer in cnn_base.layers:
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Activation):
            act_outputs.append(x)
        elif isinstance(layer, tf.keras.layers.Dropout):
            drop_outputs.append(x)

    if len(act_outputs) < 3:
        raise ValueError(f"expected >= 3 Activation layers in CNN, found {len(act_outputs)}")
    if len(drop_outputs) < 3:
        raise ValueError(f"expected >= 3 Dropout layers in CNN, found {len(drop_outputs)}")

    skip1      = act_outputs[0]   # relu after conv block 1, before MaxPool
    skip2      = act_outputs[1]   # relu after conv block 2, before MaxPool
    skip3      = act_outputs[2]   # relu after conv block 3, before MaxPool
    bottleneck = drop_outputs[2]  # dropout after 3rd MaxPool

    # decoder block 1: 32x32 -> 64x64
    xd = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='dec_up1')(bottleneck)
    xd = layers.Concatenate(name='dec_cat1')([xd, skip3])
    xd = layers.Conv2D(128, (3, 3), padding='same', name='dec_conv1')(xd)
    xd = layers.BatchNormalization(name='dec_bn1')(xd)
    xd = layers.Activation('relu', name='dec_act1')(xd)

    # decoder block 2: 64x64 -> 128x128
    xd = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='dec_up2')(xd)
    xd = layers.Concatenate(name='dec_cat2')([xd, skip2])
    xd = layers.Conv2D(64, (3, 3), padding='same', name='dec_conv2')(xd)
    xd = layers.BatchNormalization(name='dec_bn2')(xd)
    xd = layers.Activation('relu', name='dec_act2')(xd)

    # decoder block 3: 128x128 -> 256x256
    xd = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='dec_up3')(xd)
    xd = layers.Concatenate(name='dec_cat3')([xd, skip1])
    xd = layers.Conv2D(32, (3, 3), padding='same', name='dec_conv3')(xd)
    xd = layers.BatchNormalization(name='dec_bn3')(xd)
    xd = layers.Activation('relu', name='dec_act3')(xd)

    seg_maps  = layers.Conv2D(N_CLASSES, (1, 1), activation='sigmoid', name='seg_maps')(xd)
    class_out = layers.GlobalAveragePooling2D(name='class_out')(seg_maps)

    classifier = tf.keras.Model(inputs=inp, outputs=class_out, name='unet_classifier')
    segmenter  = tf.keras.Model(inputs=inp, outputs=seg_maps,  name='unet_segmenter')
    return classifier, segmenter


classifier, segmenter = build_unet_from_cnn(cnn_base, freeze_encoder=True)
classifier.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=['recall', tf.keras.metrics.AUC(name='auc', multi_label=True)],
)
trainable_params = sum(int(np.prod(v.shape)) for v in classifier.trainable_variables)
print(f"\nTrainable params: {trainable_params:,}  (encoder frozen)", flush=True)


# ---- streaming dataset (same approach as steps 1 and 2) ----
def make_dataset(X, y, sample_weight, batch_size, shuffle):
    n       = len(X)
    y_shape = (y.shape[1],) if y.ndim > 1 else ()
    sig = (
        tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=y_shape,     dtype=tf.float32),
    )
    if sample_weight is not None:
        sig += (tf.TensorSpec(shape=(), dtype=tf.float32),)

    def gen():
        while True:
            for i in range(n):
                xi = X[i].astype('float32')
                yi = y[i].astype('float32')
                if sample_weight is not None:
                    yield xi, yi, sample_weight[i].astype('float32')
                else:
                    yield xi, yi

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


any_hemorrhage = (y_train.max(axis=1) > 0).astype(int)
sample_weights = compute_sample_weight('balanced', any_hemorrhage)
train_steps    = int(np.ceil(len(x_train) / BATCH_SIZE))
val_steps      = int(np.ceil(len(x_val)   / BATCH_SIZE))
train_ds       = make_dataset(x_train, y_train, sample_weights, BATCH_SIZE, shuffle=True)
val_ds         = make_dataset(x_val,   y_val,   None,           BATCH_SIZE, shuffle=False)

os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
ckpt_path = os.path.join(MODEL_EXPORT_DIR, f'unet_checkpoint_{ts}.keras')

print(f"\nTraining U-Net classifier (frozen encoder)...", flush=True)
train_started = datetime.datetime.now()
classifier.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-6),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
    ],
)
total = datetime.datetime.now() - train_started
print(f"Training complete.", flush=True)


# ---- evaluate on test set ----
print("\nEvaluating on test set...", flush=True)
y_pred_proba = np.vstack([
    classifier.predict_on_batch(x_test[i : i + BATCH_SIZE].copy())
    for i in range(0, len(x_test), BATCH_SIZE)
])

print("\nTest Set AUC per class (U-Net):")
for i, col in enumerate(label_cols):
    try:
        auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
        print(f"  {col:25s} AUC: {auc:.4f}")
    except ValueError:
        print(f"  {col:25s} AUC: N/A (single class)")
print(f"  {'weighted average':25s} AUC: "
      f"{roc_auc_score(y_test, y_pred_proba, average='weighted'):.4f}")


# ---- save U-Net segmentation maps for a subset of positive test samples ----
# saving full-res maps for the full test set would require ~132 GB;
# limiting to N_SEG_MAPS positive samples keeps storage at ~1.3 GB (float16).
pos_idx = np.where(y_test.max(axis=1) > 0)[0][:N_SEG_MAPS]
print(f"\nGenerating U-Net seg maps for {len(pos_idx)} positive test samples...", flush=True)

unet_maps = np.vstack([
    segmenter.predict_on_batch(x_test[pos_idx[i : i + BATCH_SIZE]].copy())
    for i in range(0, len(pos_idx), BATCH_SIZE)
]).astype('float16')

seg_path = os.path.join(MODEL_EXPORT_DIR, f'unet_seg_maps_{ts}.npy')
np.save(seg_path, unet_maps)
print(f"Saved U-Net seg maps: {os.path.basename(seg_path)}  shape: {unet_maps.shape}", flush=True)
del unet_maps; gc.collect()


# ---- GradCAM helper ----
def compute_gradcam(gradcam_model, x_batch, out_size=IMG_SIZE, n_classes=N_CLASSES):
    """
    per-class GradCAM for a batch.
    tape.watch on x_tensor (a constant) lets us get gradients at an intermediate conv layer.
    returns float16 of shape (B, out_size, out_size, n_classes).
    """
    x_tensor = tf.constant(x_batch.astype('float32'))
    cls_maps = []
    for cls in range(n_classes):
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            conv_out, preds = gradcam_model(x_tensor, training=False)
            score           = tf.reduce_sum(preds[:, cls])
        grads   = tape.gradient(score, conv_out)                              # (B, h, w, C)
        weights = tf.reduce_mean(grads, axis=[1, 2])                          # (B, C)
        cam     = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)  # (B, h, w)
        cam     = tf.nn.relu(cam)
        cam     = cam / (tf.reduce_max(cam, axis=[1, 2], keepdims=True) + 1e-8)
        cam     = tf.image.resize(cam[..., tf.newaxis], [out_size, out_size])[..., 0]
        cls_maps.append(cam.numpy().astype('float16'))
    return np.stack(cls_maps, axis=-1)  # (B, out_size, out_size, n_classes)


# ---- GradCAM from CNN (step 1) ----
print("\nGenerating CNN GradCAM maps...", flush=True)
inp_g = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='cnn_gradcam_input')
xg = inp_g
last_conv_out = None
for layer in cnn_base.layers:
    xg = layer(xg)
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_out = xg
cnn_gradcam_model = tf.keras.Model(inputs=inp_g, outputs=[last_conv_out, xg])
cnn_cam_maps = np.vstack([
    compute_gradcam(cnn_gradcam_model, x_test[pos_idx[i : i + BATCH_SIZE]].copy())
    for i in range(0, len(pos_idx), BATCH_SIZE)
])
cnn_cam_path = os.path.join(MODEL_EXPORT_DIR, f'cnn_gradcam_maps_{ts}.npy')
np.save(cnn_cam_path, cnn_cam_maps)
print(f"Saved CNN GradCAM maps: {os.path.basename(cnn_cam_path)}  shape: {cnn_cam_maps.shape}",
      flush=True)
del cnn_cam_maps, cnn_gradcam_model; gc.collect()


# ---- GradCAM from ViT (step 2) ----
# the ViT's only 2D spatial layer is the patch embedding Conv2D, so GradCAM
# targets that layer and upsamples from (16, 16) back to (256, 256).
print("\nGenerating ViT GradCAM maps...", flush=True)
if not os.path.exists(VIT_CKPT):
    print(f"WARNING: ViT checkpoint not found ({VIT_CKPT}) -- skipping ViT GradCAM.", flush=True)
else:
    print(f"Loading ViT checkpoint: {os.path.basename(VIT_CKPT)}", flush=True)

    class PositionEmbedding(tf.keras.layers.Layer):
        def __init__(self, num_patches, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.num_patches = num_patches
            self.embed       = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

        def call(self, x):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            return x + self.embed(positions)

        def get_config(self):
            config = super().get_config()
            config.update({'num_patches': self.num_patches,
                           'embed_dim':   self.embed.output_dim})
            return config

    vit_base = tf.keras.models.load_model(
        VIT_CKPT, custom_objects={'PositionEmbedding': PositionEmbedding}
    )
    vit_conv_layers   = [l for l in vit_base.layers if isinstance(l, tf.keras.layers.Conv2D)]
    vit_gradcam_model = tf.keras.Model(
        inputs=vit_base.input,
        outputs=[vit_conv_layers[0].output, vit_base.output],
    )
    vit_cam_maps = np.vstack([
        compute_gradcam(vit_gradcam_model, x_test[pos_idx[i : i + BATCH_SIZE]].copy())
        for i in range(0, len(pos_idx), BATCH_SIZE)
    ])
    vit_cam_path = os.path.join(MODEL_EXPORT_DIR, f'vit_gradcam_maps_{ts}.npy')
    np.save(vit_cam_path, vit_cam_maps)
    print(f"Saved ViT GradCAM maps: {os.path.basename(vit_cam_path)}  shape: {vit_cam_maps.shape}",
          flush=True)
    del vit_cam_maps, vit_base, vit_gradcam_model; gc.collect()


# ---- GradCAM from Hybrid ----
print("\nGenerating Hybrid GradCAM maps...", flush=True)
if not os.path.exists(HYBRID_CKPT):
    print(f"WARNING: Hybrid checkpoint not found ({HYBRID_CKPT}) -- skipping Hybrid GradCAM.", flush=True)
else:
    # hybrid = (CNN + ViT) / 2, so hybrid GradCAM = average of CNN and ViT GradCAM.
    # the 1/2 scale factor cancels after per-map normalisation, making this exact.
    # loading the hybrid model as a Keras model and extracting internal Conv2D layers
    # is not viable because the CNN/ViT are stored as nested sub-model layers.
    hybrid_cam_maps = ((np.load(cnn_cam_path).astype('float32') +
                        np.load(vit_cam_path).astype('float32')) / 2).astype('float16')
    hybrid_cam_path = os.path.join(MODEL_EXPORT_DIR, f'hybrid_gradcam_maps_{ts}.npy')
    np.save(hybrid_cam_path, hybrid_cam_maps)
    print(f"Saved Hybrid GradCAM maps: {os.path.basename(hybrid_cam_path)}  shape: {hybrid_cam_maps.shape}",
          flush=True)
    del hybrid_cam_maps; gc.collect()

# save the test-set indices so maps can be matched back to samples
idx_path = os.path.join(MODEL_EXPORT_DIR, f'seg_maps_test_idx_{ts}.npy')
np.save(idx_path, pos_idx)
print(f"Saved test sample indices: {os.path.basename(idx_path)}", flush=True)


# ---- save model and log ----
os.chdir(os.path.expanduser('~/thesis-xai'))

dummy_grid = types.SimpleNamespace(best_params_={
    'clf__batch_size': BATCH_SIZE,
    'clf__epochs':     EPOCHS,
    'clf__model__lr':  LR,
}, best_score_=float('nan'))

_, keras_path = save_models(dummy_grid, keras_model=classifier, method='unet')

log_run(
    grid=dummy_grid,
    y_test=y_test,
    y_pred_proba=y_pred_proba,
    x_train=x_train,
    x_test=x_test,
    label_cols=label_cols,
    N_TRAIN_PATIENTS=meta['N_TRAIN_PATIENTS'],
    N_TEST_PATIENTS=meta['N_TEST_PATIENTS'],
    total_timedelta=total,
    model_pkl=keras_path.with_suffix('.pkl'),
    model_h5=keras_path,
    method='unet',
    loss='bce',
    source_cnn=os.path.basename(CNN_CKPT),
    source_vit=os.path.basename(VIT_CKPT),
)

print("\nDone!")
