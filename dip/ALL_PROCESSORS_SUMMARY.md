# Digital Image Processing (DIP) Processors - Complete Summary

This document provides a comprehensive overview of all DIP processors available in the LeRobot recording pipeline.

## 📋 Complete Processor List

### **Dim Lighting Enhancement**
1. **CLAHE Processor** - Contrast Limited Adaptive Histogram Equalization
2. **Gamma Processor** - Gamma correction enhancement

### **Shadow Removal**
3. **Shadow Removal Processor** - Adaptive illumination correction
4. **SSR Processor** - Single Scale Retinex

### **Deblurring (Partial Occlusion)**
5. **Wiener Processor** - Wiener deconvolution deblurring
6. **Richardson-Lucy Processor** - Richardson-Lucy deconvolution

---

## 📁 File Structure

```
dip/
├── dim_lighting/
│   ├── clahe_processor.py          ✓ CLAHE enhancement
│   ├── gamma_processor.py          ✓ Gamma correction
│   ├── clahe_enhancement.py        (training script)
│   ├── gamma_enhancement.py        (training script)
│   └── output/
│       ├── clahe_params.json
│       └── gamma_params.json
│
├── shadow_removal/
│   ├── shadow_removal_processor.py ✓ Adaptive illumination
│   ├── ssr_processor.py            ✓ Single Scale Retinex
│   ├── shadow_comparison.py        (training script)
│   └── ssr_enhancement.py          (training script)
│
└── partial_occlusion/
    ├── wiener_processor.py         ✓ Wiener deconvolution
    ├── richardson_lucy_processor.py ✓ Richardson-Lucy
    ├── weiner.py                   (standalone script)
    ├── lrd.py                      (standalone script)
    └── test_deblur.py              (sanity check)
```

---

## 🚀 Usage Guide

### Available DIP Methods

Use `--dip.method=<method>` to select a processor:

| Method | Purpose | Typical Suffix |
|--------|---------|----------------|
| `clahe` | CLAHE enhancement for dim lighting | `_dim` |
| `gamma_enhancement` | Gamma correction for dim lighting | `_dim` |
| `shadow_removal` | Adaptive illumination for shadows | `_shadow` |
| `ssr_enhancement` | SSR for shadow removal | `_shadow` |
| `wiener_deblur` | Wiener deconvolution for blur | `_blurred` |
| `richardson_lucy_deblur` | Richardson-Lucy for blur | `_blurred` |
| `both` | CLAHE + Shadow Removal | `_dim` |
| `none` | No processing | - |

### Example Commands

#### 1. CLAHE Enhancement
```bash
python lerobot_record_dip.py \
    --robot.type=so100_follower \
    --dataset.repo_id=username/dataset \
    --dip.method=clahe \
    --dip.camera_suffix=_dim
```

#### 2. Gamma Enhancement
```bash
python lerobot_record_dip.py \
    --robot.type=so100_follower \
    --dataset.repo_id=username/dataset \
    --dip.method=gamma_enhancement \
    --dip.camera_suffix=_dim
```

#### 3. Shadow Removal
```bash
python lerobot_record_dip.py \
    --robot.type=so100_follower \
    --dataset.repo_id=username/dataset \
    --dip.method=shadow_removal \
    --dip.camera_suffix=_shadow
```

#### 4. SSR Enhancement
```bash
python lerobot_record_dip.py \
    --robot.type=so100_follower \
    --dataset.repo_id=username/dataset \
    --dip.method=ssr_enhancement \
    --dip.camera_suffix=_shadow
```

#### 5. Wiener Deblurring
```bash
python lerobot_record_dip.py \
    --robot.type=so100_follower \
    --dataset.repo_id=username/dataset \
    --dip.method=wiener_deblur \
    --dip.camera_suffix=_blurred
```

#### 6. Richardson-Lucy Deblurring
```bash
python lerobot_record_dip.py \
    --robot.type=so100_follower \
    --dataset.repo_id=username/dataset \
    --dip.method=richardson_lucy_deblur \
    --dip.camera_suffix=_blurred
```

---

## ⚙️ Processor Details

### 1. CLAHE Processor
- **File**: `dim_lighting/clahe_processor.py`
- **Parameters**: Loaded from `clahe_params.json`
  - `clip_limit`: Contrast limiting threshold
  - `tile_grid_size`: Grid size for histogram equalization
- **Use Case**: Enhance dim/low-light images
- **Training**: Run `clahe_enhancement.py` to optimize parameters

### 2. Gamma Processor
- **File**: `dim_lighting/gamma_processor.py`
- **Parameters**: Loaded from `gamma_params.json`
  - `gamma`: Gamma correction value (< 1 brightens, > 1 darkens)
- **Use Case**: Brighten dim images using gamma correction
- **Training**: Run `gamma_enhancement.py` to optimize parameters

### 3. Shadow Removal Processor
- **File**: `shadow_removal/shadow_removal_processor.py`
- **Parameters**: Learned from reference images
  - `target_luminance`: Target mean luminance value
  - `sigma`: Gaussian blur sigma (default: 50)
- **Use Case**: Remove shadows using adaptive illumination
- **Training**: Automatically learns from reference images

### 4. SSR Processor
- **File**: `shadow_removal/ssr_processor.py`
- **Parameters**: Fixed parameters
  - `sigma`: Gaussian blur sigma (default: 50)
  - `gain`: Output gain factor (default: 0.6)
- **Use Case**: Remove shadows using Single Scale Retinex
- **Training**: Uses fixed parameters, no training needed

### 5. Wiener Processor
- **File**: `partial_occlusion/wiener_processor.py`
- **Parameters**: Fixed parameters
  - `kernel_size`: Blur kernel size (default: 15)
  - `sigma`: Gaussian sigma (default: 3.0)
  - `k`: Noise-to-signal ratio (default: 0.01)
- **Use Case**: Deblur images using Wiener deconvolution
- **Training**: Uses fixed parameters, no training needed
- **Speed**: Fast (FFT-based)

### 6. Richardson-Lucy Processor
- **File**: `partial_occlusion/richardson_lucy_processor.py`
- **Parameters**: Fixed parameters
  - `kernel_size`: PSF kernel size (default: 15)
  - `sigma`: Gaussian sigma (default: 5.0)
  - `iterations`: Number of iterations (default: 30)
- **Use Case**: Blind deblurring using Richardson-Lucy
- **Training**: Uses fixed parameters, no training needed
- **Speed**: Slower (iterative algorithm)
- **Requires**: scikit-image (`pip install scikit-image`)

---

## 🔧 Configuration Options

All processors support these configuration options:

```bash
--dip.method=<method>              # Processor to use
--dip.camera_suffix=<suffix>       # Camera suffix (optional)
--dip.clahe_params_file=<path>     # Custom CLAHE params (optional)
--dip.gamma_params_file=<path>     # Custom gamma params (optional)
```

### Camera Suffix Behavior

**Without suffix** (`--dip.camera_suffix=""` or omit):
- Processes cameras matching parameter file names exactly
- Example: `side`, `front`

**With suffix** (`--dip.camera_suffix=_dim`):
- Processes cameras ending with the suffix
- Removes suffix to find parameters
- Example: `side_dim` → uses `side` parameters

---

## 📊 Performance Comparison

| Processor | Speed | Quality | Parameters | Dependencies |
|-----------|-------|---------|------------|--------------|
| CLAHE | Fast | Good | Learned | OpenCV |
| Gamma | Very Fast | Good | Learned | OpenCV |
| Shadow Removal | Fast | Very Good | Learned | OpenCV |
| SSR | Fast | Good | Fixed | OpenCV |
| Wiener | Fast | Good | Fixed | OpenCV, NumPy |
| Richardson-Lucy | Slow | Very Good | Fixed | scikit-image |

---

## ✅ Sanity Check Results

**Test Status** (from `test_deblur.py`):
- ✓ Wiener deconvolution: **WORKING**
- ✓ Richardson-Lucy deconvolution: **WORKING**

**Test Outputs**:
- `output/test_blurred.jpg` - Artificially blurred test image
- `output/test_wiener_deblurred.jpg` - Wiener result
- `output/test_rl_deblurred.jpg` - Richardson-Lucy result
- `output/comparison_deblur_methods.jpg` - Side-by-side comparison

---

## 🎯 Recommendations

### For Dim Lighting:
- **Best quality**: CLAHE (optimized per camera)
- **Fastest**: Gamma correction
- **Most flexible**: CLAHE with custom parameters

### For Shadow Removal:
- **Best quality**: Shadow Removal (adaptive illumination)
- **Fastest**: SSR (Single Scale Retinex)
- **Most robust**: Shadow Removal

### For Deblurring:
- **Best quality**: Richardson-Lucy (iterative, slower)
- **Fastest**: Wiener (FFT-based)
- **Best for real-time**: Wiener

---

## 📝 Notes

1. **Camera Suffix is Optional**: You can omit it to process all cameras
2. **Multiple Processors**: Currently only one method at a time (except `both` for CLAHE+Shadow)
3. **Parameter Files**: CLAHE and Gamma require training to generate parameter files
4. **Dependencies**: Richardson-Lucy requires scikit-image installation
5. **Processing Order**: Processors are applied in the order they're added to the pipeline

---

## 🔄 Workflow

1. **Collect Training Data**: Capture images with and without the issue (dim/shadow/blur)
2. **Train Parameters**: Run enhancement scripts to optimize parameters
3. **Configure Recording**: Set `--dip.method` and `--dip.camera_suffix`
4. **Record Dataset**: Run `lerobot_record_dip.py` with DIP enabled
5. **Verify Results**: Check processed images during recording

---

## 📚 All Available Processors Summary

| # | Processor | Method Name | Category | Status |
|---|-----------|-------------|----------|--------|
| 1 | CLAHE | `clahe` | Dim Lighting | ✅ Ready |
| 2 | Gamma | `gamma_enhancement` | Dim Lighting | ✅ Ready |
| 3 | Shadow Removal | `shadow_removal` | Shadow | ✅ Ready |
| 4 | SSR | `ssr_enhancement` | Shadow | ✅ Ready |
| 5 | Wiener | `wiener_deblur` | Deblurring | ✅ Ready |
| 6 | Richardson-Lucy | `richardson_lucy_deblur` | Deblurring | ✅ Ready |

**Total: 6 processors across 3 categories** 🎉
