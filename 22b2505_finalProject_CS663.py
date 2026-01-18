# Name: Afnan Abdul Gafoor  
# Roll No: 22B2505  
# Please refer to the attached presentation slides for detailed methodology and references.


import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def bilinear_interpolate(img, y, x):
    h, w = img.shape
    
    y_flat = y.ravel()
    x_flat = x.ravel()
    
    x0 = np.floor(x_flat).astype(int)
    y0 = np.floor(y_flat).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]
    
    wa = (x1 - x_flat) * (y1 - y_flat)
    wb = (x1 - x_flat) * (y_flat - y0)
    wc = (x_flat - x0) * (y1 - y_flat)
    wd = (x_flat - x0) * (y_flat - y0)
    
    result = wa * Ia + wb * Ib + wc * Ic + wd * Id
    
    out_of_bounds = (x_flat < 0) | (x_flat >= w) | (y_flat < 0) | (y_flat >= h)
    result[out_of_bounds] = 0.0
    
    return result.reshape(y.shape)


class LBPOperator:
    def __init__(self, P=8, R=1.0):
        self.P = P
        self.R = R
        self.neighbors = self._get_circular_neighbors()
        self._ri_lut_cache = None
        self._riu2_lut_cache = None

    def _get_circular_neighbors(self):
        angles = 2 * np.pi * np.arange(self.P) / self.P
        neighbors = np.zeros((self.P, 2))
        neighbors[:, 0] = -self.R * np.sin(angles)  # y coordinates
        neighbors[:, 1] = self.R * np.cos(angles)   # x coordinates
        return neighbors

    @property
    def ri_lut(self):
        if self._ri_lut_cache is None:
            self._ri_lut_cache = self._create_rotation_invariant_lut()
        return self._ri_lut_cache

    @property
    def riu2_lut(self):
        if self._riu2_lut_cache is None:
            self._riu2_lut_cache = self._create_riu2_lookup_table()
        return self._riu2_lut_cache

    def _rotate_pattern(self, pattern, shifts):
        mask = (1 << self.P) - 1
        return ((pattern >> shifts) | (pattern << (self.P - shifts))) & mask # circular bit-wise right shift

    def _create_rotation_invariant_lut(self):
        lut = np.zeros(2**self.P, dtype=np.int32)
        for i in range(2**self.P):
            min_val = i
            for shift in range(1, self.P):
                rotated = self._rotate_pattern(i, shift)
                if rotated < min_val:
                    min_val = rotated
            lut[i] = min_val
        return lut

    def _uniformity_measure(self, pattern):
        binary = [(pattern >> i) & 1 for i in range(self.P)]
        binary.append(binary[0])
        transitions = 0
        for i in range(self.P):
            if binary[i] != binary[i+1]:
                transitions += 1
        return transitions

    def _create_riu2_lookup_table(self):
        lut = np.zeros(2**self.P, dtype=np.int32)
        misc_label = self.P + 1
        for pattern in range(2**self.P):
            ri_pattern = self.ri_lut[pattern]
            u = self._uniformity_measure(ri_pattern)
            if u <= 2:
                ones = bin(ri_pattern).count('1')
                lut[pattern] = ones
            else:
                lut[pattern] = misc_label
        return lut

    def compute_lbp_basic(self, img):
        h, w = img.shape
        lbp = np.zeros((h, w), dtype=np.int32)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        for p in range(self.P):
            ny = yy + self.neighbors[p, 0]
            nx = xx + self.neighbors[p, 1]
            
            neighbor_vals = bilinear_interpolate(img, ny, nx)
            
            lbp += ((neighbor_vals >= img).astype(np.uint32) << p)
        border = int(np.ceil(self.R))
        if border > 0:
            lbp[:border, :] = 0
            lbp[-border:, :] = 0
            lbp[:, :border] = 0
            lbp[:, -border:] = 0
        return lbp

    def compute_lbp_riu2(self, img):
        lbp_basic = self.compute_lbp_basic(img)
        lbp_riu2 = self.riu2_lut[lbp_basic]
        return lbp_riu2

    def compute_variance(self, img):
        h, w = img.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        neighbor_stack = np.zeros((self.P, h, w), dtype=np.float32)
        for p in range(self.P):
            ny = yy + self.neighbors[p, 0]
            nx = xx + self.neighbors[p, 1]
            
            neighbor_vals = bilinear_interpolate(img, ny, nx)
            neighbor_stack[p] = neighbor_vals
            
        var = np.var(neighbor_stack, axis=0)
        border = int(np.ceil(self.R))
        if border > 0:
            var[:border, :] = 0
            var[-border:, :] = 0
            var[:, :border] = 0
            var[:, -border:] = 0
        return var

    def compute_histogram(self, lbp_img, normalize=True):
        max_label = int(lbp_img.max())
        if max_label <= self.P + 1:
            num_bins = self.P + 2
        else:
            num_bins = 2**self.P
        hist, _ = np.histogram(lbp_img.ravel(), bins=num_bins, range=(0, num_bins))
        if normalize:
            hist = hist.astype(np.float64)
            hist = hist / (hist.sum() + 1e-10)
        return hist

    def compute_joint_histogram(self, lbp_img, var_img, var_cutpoints=None, var_bins=10, normalize=True):
        valid_mask = lbp_img >= 0
        lbp_valid = lbp_img[valid_mask].ravel()
        var_valid = var_img[valid_mask].ravel()
        nz_mask = var_valid > 0
        lbp_valid = lbp_valid[nz_mask]
        var_valid = var_valid[nz_mask]
        lbp_max_label = int(lbp_img.max())
        if lbp_max_label <= self.P + 1:
            n_lbp_bins = self.P + 2
        else:
            n_lbp_bins = 2**self.P
        if var_cutpoints is None:
            vmin, vmax = var_valid.min(), var_valid.max()
            if vmin == vmax:
                var_quantized = np.zeros_like(var_valid, dtype=int)
            else:
                edges = np.linspace(vmin, vmax, var_bins + 1)
                var_quantized = np.digitize(var_valid, edges[1:-1])
        else:
            edges = np.asarray(var_cutpoints)
            var_quantized = np.digitize(var_valid, edges)
        lbp_valid = lbp_valid.astype(int)
        var_quantized = np.clip(var_quantized.astype(int), 0, var_bins - 1)
        valid_indices = (0 <= lbp_valid) & (lbp_valid < n_lbp_bins)
        lbp_valid = lbp_valid[valid_indices]
        var_quantized = var_quantized[valid_indices]
        flat_indices = lbp_valid * var_bins + var_quantized
        hist_flat = np.bincount(flat_indices, minlength=n_lbp_bins * var_bins)
        hist_flat = hist_flat.astype(np.float64)
        if normalize:
            s = hist_flat.sum()
            if s > 0:
                hist_flat = hist_flat / s
        return hist_flat

def compute_var_cutpoints_for_operator(patches_list, op, var_bins=10):
    pooled_vars = []
    batch_size = 100
    for i in range(0, len(patches_list), batch_size):
        batch = patches_list[i:i+batch_size]
        for patch in batch:
            varmap = op.compute_variance(patch)
            vals = varmap[varmap > 0].ravel()
            if vals.size:
                pooled_vars.append(vals)
    if len(pooled_vars) == 0:
        return np.array([])
    pooled = np.concatenate(pooled_vars)
    qs = [100.0 * i / var_bins for i in range(1, var_bins)]
    cutpoints = np.percentile(pooled, qs)
    return cutpoints

class TextureClassifier:

    def __init__(self, operators):
        self.operators = operators if isinstance(operators, list) else [operators]
        self.models = {}
        self.var_cutpoints_per_op = []
        self.use_variance = False
        self.var_bins = 10

    def extract_features(self, img, use_variance=False, var_bins=10, var_cutpoints_list=None):
        if var_cutpoints_list is None:
            var_cutpoints_list = [None] * len(self.operators)
        features = []
        for idx, op in enumerate(self.operators):
            lbp = op.compute_lbp_riu2(img)
            if use_variance:
                var = op.compute_variance(img)
                hist = op.compute_joint_histogram(
                    lbp, var,
                    var_cutpoints=var_cutpoints_list[idx],
                    var_bins=var_bins,
                    normalize=True
                )
            else:
                hist = op.compute_histogram(lbp, normalize=True)
            features.append(hist)
        return features

    def extract_features_batch(self, images, use_variance=False, var_bins=10, var_cutpoints_list=None):
        return [self.extract_features(img, use_variance, var_bins, var_cutpoints_list)
                for img in images]

    def train(self, images_by_class, use_variance=False, var_bins=10):
        self.models = {}
        self.use_variance = use_variance
        self.var_bins = var_bins

        # compute var cutpoints if needed
        if use_variance:
            all_patches = []
            for class_name, images in images_by_class.items():
                all_patches.extend(images)
            self.var_cutpoints_per_op = []
            for idx, op in enumerate(self.operators):
                cutpoints = compute_var_cutpoints_for_operator(all_patches, op, var_bins)
                self.var_cutpoints_per_op.append(cutpoints)
        else:
            self.var_cutpoints_per_op = [None] * len(self.operators)

        # build model histograms
        for class_name, images in images_by_class.items():
            all_features = self.extract_features_batch(
                images, use_variance, var_bins,
                var_cutpoints_list=self.var_cutpoints_per_op
            )
            accumulated_features = None
            for features in all_features:
                if accumulated_features is None:
                    accumulated_features = [np.zeros_like(f) for f in features]
                for i, f in enumerate(features):
                    accumulated_features[i] += f
            count = len(images) if len(images) > 0 else 1
            for i in range(len(accumulated_features)):
                accumulated_features[i] /= count
                accumulated_features[i] += 1e-9
                accumulated_features[i] /= accumulated_features[i].sum()
            self.models[class_name] = accumulated_features

    def log_likelihood(self, sample_hist, model_hist):
        model_hist = model_hist + 1e-10
        sample_hist = sample_hist + 1e-10
        model_hist = model_hist / model_hist.sum()
        sample_hist = sample_hist / sample_hist.sum()
        return np.sum(sample_hist * np.log(model_hist))

    def classify_single(self, img, use_variance=None):
        if use_variance is None:
            use_variance = self.use_variance
        var_bins = getattr(self, 'var_bins', 10)
        sample_features = self.extract_features(
            img, use_variance, var_bins,
            var_cutpoints_list=self.var_cutpoints_per_op
        )
        scores = {}
        for class_name, model_features in self.models.items():
            score = 0.0
            for sample_hist, model_hist in zip(sample_features, model_features):
                score += self.log_likelihood(sample_hist, model_hist)
            scores[class_name] = score
        predicted_class = max(scores, key=scores.get)
        return predicted_class, scores

def extract_subimages(img, size=16, stride=None):
    if stride is None:
        stride = size
    h, w = img.shape
    subimages = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            subimg = img[y:y+size, x:x+size]
            subimages.append(subimg)
    return subimages

def classify_full_image_by_patches(classifier, full_img, patch_size=16, stride=None, use_variance=False):
    patches = extract_subimages(full_img, size=patch_size, stride=stride)
    if len(patches) == 0:
        return classifier.classify_single(full_img, use_variance)
    per_patch_features = classifier.extract_features_batch(
        patches,
        use_variance=use_variance,
        var_bins=classifier.var_bins,
        var_cutpoints_list=classifier.var_cutpoints_per_op
    )
    class_scores = {c: 0.0 for c in classifier.models.keys()}
    for feat_list in per_patch_features:
        for class_name, model_feat_list in classifier.models.items():
            s = 0.0
            for sample_hist, model_hist in zip(feat_list, model_feat_list):
                s += classifier.log_likelihood(sample_hist, model_hist)
            class_scores[class_name] += s
    pred = max(class_scores.items(), key=lambda kv: kv[1])[0]
    return pred, class_scores

def load_texture_images(base_dir, texture_names, angles=[0]):
    images = {}
    for texture in texture_names:
        for angle in angles:
            filename = f"{texture}.{angle:03d}.tiff"
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath):
                img = Image.open(filepath).convert('L')
                images[(texture, angle)] = np.array(img, dtype=np.float32)
            else:
                print(f"Warning: Could not find {filepath}")
    return images

RESULTS_ROOT = "22b2505_results"

def ensure_dirs():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_ROOT, "lbp_maps"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_ROOT, "histograms"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_ROOT, "confusion_matrices"), exist_ok=True)

def img_for_display(arr):
    a = np.array(arr, dtype=np.float32)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return (255.0 * a).astype(np.uint8)

def save_figure(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def visualize_lbp_maps(op, img, prefix):
    lbp_basic = op.compute_lbp_basic(img)
    lbp_riu2 = op.compute_lbp_riu2(img)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))
    plt.subplots_adjust(top=0.90)

    axs[0].imshow(img_for_display(img), cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(img_for_display(lbp_basic), cmap='gray')
    axs[1].set_title(f"LBP basic (P={op.P})")
    axs[1].axis('off')

    axs[2].imshow(img_for_display(lbp_riu2), cmap='gray')
    axs[2].set_title("LBP riu2")
    axs[2].axis('off')

    out = os.path.join(RESULTS_ROOT, "lbp_maps", f"{prefix}.png")
    save_figure(fig, out)

def visualize_histograms(op, img, prefix, var_bins=10):
    lbp = op.compute_lbp_riu2(img)
    hist = op.compute_histogram(lbp, normalize=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.bar(np.arange(hist.size), hist)
    ax.set_title("LBP riu2 histogram")
    ax.set_xlabel("riu2 bin")
    ax.set_ylabel("Normalized frequency")
    out = os.path.join(RESULTS_ROOT, "histograms", f"{prefix}_riu2_hist.png")
    save_figure(fig, out)

    # small joint heatmap saved as separate file only if VAR produces non-empty data
    try:
        var = op.compute_variance(img)
        if np.any(var > 0):
            # compute var cutpoints locally for visualization only
            vmin, vmax = var[var > 0].min(), var[var > 0].max()
            if vmin == vmax:
                edges = None
            else:
                edges = np.linspace(vmin, vmax, var_bins + 1)
            joint = op.compute_joint_histogram(lbp, var, var_cutpoints=edges if edges is not None else None, var_bins=var_bins, normalize=True)
            n_lbp_bins = op.P + 2 if lbp.max() <= op.P + 1 else 2**op.P
            joint = joint.reshape((n_lbp_bins, var_bins))
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            im = ax.imshow(joint, aspect='auto', origin='lower')
            ax.set_title("Joint LBPriu2 x VAR (heatmap)")
            ax.set_xlabel("VAR bin")
            ax.set_ylabel("LBP riu2 bin")
            fig.colorbar(im, ax=ax)
            out = os.path.join(RESULTS_ROOT, "histograms", f"{prefix}_joint.png")
            save_figure(fig, out)
    except Exception:
        pass


def save_confusion_matrix(confusion, texture_names, prefix):
    n = len(texture_names)
    matrix = np.zeros((n, n), dtype=int)
    name_to_idx = {n: i for i, n in enumerate(texture_names)}
    for true in texture_names:
        for pred in texture_names:
            matrix[name_to_idx[true], name_to_idx[pred]] = confusion[true][pred]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    ax.set_title(f"Confusion matrix ({prefix})")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(texture_names, rotation=90)
    ax.set_yticklabels(texture_names)
    fig.colorbar(im, ax=ax)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, matrix[i, j], ha='center', va='center', color='black', fontsize=8)
    out = os.path.join(RESULTS_ROOT, "confusion_matrices", f"{prefix}_confusion.png")
    save_figure(fig, out)

def run_experiment_1(rotate_dir='rotate'):
    texture_names = [
        'bark', 'brick', 'bubbles', 'grass', 'leather',
        'pigskin', 'raffia', 'sand', 'straw', 'water',
        'weave', 'wood', 'wool'
    ]

    all_angles = [0, 30, 60, 90, 120, 150, 200]

    print("Loading texture images...")
    images = load_texture_images(rotate_dir, texture_names, all_angles)

    training_angles = [0, 30, 60, 90]

    operators_configs = [
        ([LBPOperator(8, 1.0)], "LBP_8,1", False),
        ([LBPOperator(16, 2.0)], "LBP_16,2", False),
        ([LBPOperator(24, 3.0)], "LBP_24,3", False),
        ([LBPOperator(16, 2.0)], "LBP_16,2/VAR", True),
        ([LBPOperator(24, 3.0)], "LBP_24,3/VAR", True),
        ([LBPOperator(8, 1.0), LBPOperator(16, 2.0)], "LBP_8,1+16,2", False),
    ]

    results = {}

    for operators, op_name, use_var in operators_configs:
        print("\n" + "="*60)
        print(f"Testing operator: {op_name}")
        print("="*60)

        confusion = {t: {t2: 0 for t2 in texture_names} for t in texture_names}
        results[op_name] = {}

        for train_angle in training_angles:
            print(f"\nTraining with rotation angle: {train_angle}°")

            # prepare training data (16x16 subimages)
            training_data = {}
            for texture in texture_names:
                key = (texture, train_angle)
                if key in images:
                    subimages = extract_subimages(images[key], size=16)
                    training_data[texture] = subimages

            # Train classifier (with variance option)
            classifier = TextureClassifier(operators)
            classifier.train(training_data, use_variance=use_var, var_bins=10)

            # Test on full images (all angles except train_angle)
            test_angles = [a for a in all_angles if a != train_angle]
            total_correct = 0
            total_samples = 0

            print("  Testing...", end='', flush=True)
            for test_angle in test_angles:
                for texture in texture_names:
                    key = (texture, test_angle)
                    if key in images:
                        predicted, _ = classify_full_image_by_patches(
                            classifier, images[key], patch_size=16, use_variance=use_var
                        )
                        confusion[texture][predicted] += 1
                        total_samples += 1
                        if predicted == texture:
                            total_correct += 1

                        prefix_base = f"{op_name}_train{train_angle}_{texture}_{test_angle}"
                        for oi, op in enumerate(operators):
                            prefix = prefix_base + f"_op{oi}_P{op.P}R{op.R}"
                            visualize_lbp_maps(op, images[key], prefix)
                            visualize_histograms(op, images[key], prefix)

            acc = 100 * total_correct / total_samples if total_samples > 0 else 0
            results[op_name][train_angle] = acc
            print(f" done. Accuracy: {acc:.2f}% ({total_correct}/{total_samples})")

        safe_name = op_name.replace("/", "_")
        save_confusion_matrix(confusion, texture_names, prefix=safe_name)

    return results

if __name__ == "__main__":
    import time
    ensure_dirs()

    if not os.path.exists('rotate'):
        print("Error: 'rotate' directory not found!")
        print("Please ensure the rotated texture images are in a 'rotate' directory.")
        exit(1)

    print("\n" + "="*80)
    print("Running Full Experiment (13 textures)")
    print("="*80)
    start = time.time()
    results = run_experiment_1()
    print(f"\nExperiment completed in {time.time() - start:.2f} seconds")

    summary_path = os.path.join(RESULTS_ROOT, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Experiment 1 results (accuracy % by operator and training angle)\n")
        for op_name, d in results.items():
            f.write(f"\nOperator: {op_name}\n")
            for angle, acc in d.items():
                f.write(f"  Train angle {angle}: {acc:.2f}%\n")
            f.write(f"  Avg: {np.mean(list(d.values())):.2f}%\n")

    print(f"\nSummary written to {summary_path}")