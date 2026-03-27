import os
import shutil
import random
import glob

def split_weather_dataset(
        base_dir,
        output_dir=None,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    random.seed(random_seed)

    if output_dir is None:
        output_dir = base_dir

    ##to create the datasettt

    for phase in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, phase), exist_ok=True)

    classes = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d not in ["train", "val", "test"]
    ]

    print("YAY!", classes)

    for c in classes:
        c_dir = os.path.join(base_dir, c)
        images = [
            f for f in os.listdir(c_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        n = len(images)
        train_end = int(n*train_ratio)
        val_end = train_end + int(n*val_ratio)

        train_images = images[:train_end]
        val_images   = images[train_end:val_end]
        test_images  = images[val_end:]


        for img in train_images:
            os.makedirs(os.path.join(output_dir, "train", c), exist_ok=True)
            shutil.copy(
                os.path.join(c_dir, img),
                os.path.join(output_dir, "train", c, img)
            )

        for img in test_images:
            os.makedirs(os.path.join(output_dir, "test", c), exist_ok=True)
            shutil.copy(
                os.path.join(c_dir, img),
                os.path.join(output_dir, "test", c, img)
            )

        for img in val_images:
            os.makedirs(os.path.join(output_dir, "val", c), exist_ok=True)
            shutil.copy(
                os.path.join(c_dir, img),
                os.path.join(output_dir, "val", c, img)
            )

        print(f"Class '{c}': total {n} → train {len(train_images)}, val {len(val_images)}, test {len(test_images)}")

if __name__ == "__main__":
    split_weather_dataset(
        base_dir="dataset",
        output_dir="dataset",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    