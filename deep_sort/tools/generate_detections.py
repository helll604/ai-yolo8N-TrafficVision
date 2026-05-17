# vim: expandtab:ts=4:sw=4

import os
import errno
import argparse

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

physical_devices = tf.config.experimental.list_physical_devices("GPU")

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(
        physical_devices[0],
        True
    )


def _run_in_batches(function, data_dict, out, batch_size):

    data_len = len(out)
    num_batches = int(data_len / batch_size)

    start, end = 0, 0

    for i in range(num_batches):

        start = i * batch_size
        end = (i + 1) * batch_size

        batch_data = {
            key: value[start:end]
            for key, value in data_dict.items()
        }

        out[start:end] = function(batch_data)

    if end < len(out):

        batch_data = {
            key: value[end:]
            for key, value in data_dict.items()
        }

        out[end:] = function(batch_data)


def extract_image_patch(image, bbox, patch_shape):

    bbox = np.array(bbox)

    if patch_shape is not None:

        target_aspect = (
            float(patch_shape[1]) / patch_shape[0]
        )

        new_width = target_aspect * bbox[3]

        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    bbox[2:] += bbox[:2]

    bbox = bbox.astype(int)

    bbox[:2] = np.maximum(0, bbox[:2])

    bbox[2:] = np.minimum(
        np.asarray(image.shape[:2][::-1]) - 1,
        bbox[2:]
    )

    if np.any(bbox[:2] >= bbox[2:]):
        return None

    sx, sy, ex, ey = bbox

    image = image[sy:ey, sx:ex]

    image = cv2.resize(
        image,
        tuple(patch_shape[::-1])
    )

    return image


class ImageEncoder(object):

    def __init__(
        self,
        checkpoint_filename,
        input_name="images",
        output_name="features"
    ):

        self.session = tf.Session()

        with tf.gfile.GFile(
            checkpoint_filename,
            "rb"
        ) as file_handle:

            graph_def = tf.GraphDef()

            graph_def.ParseFromString(
                file_handle.read()
            )

        tf.import_graph_def(
            graph_def,
            name=""
        )

        graph = tf.get_default_graph()

        tensor_names = [
            tensor.name
            for tensor in graph.as_graph_def().node
        ]

        input_tensor = None
        output_tensor = None

        for name in tensor_names:

            if "images" in name.lower():
                input_tensor = name + ":0"

            if "features" in name.lower():
                output_tensor = name + ":0"

        if input_tensor is None:
            input_tensor = "images:0"

        if output_tensor is None:
            output_tensor = "features:0"

        self.input_var = graph.get_tensor_by_name(
            input_tensor
        )

        self.output_var = graph.get_tensor_by_name(
            output_tensor
        )

        self.feature_dim = (
            self.output_var.get_shape().as_list()[-1]
        )

        self.image_shape = (
            self.input_var.get_shape().as_list()[1:]
        )

    def __call__(self, data_x, batch_size=32):

        out = np.zeros(
            (len(data_x), self.feature_dim),
            np.float32
        )

        _run_in_batches(
            lambda x: self.session.run(
                self.output_var,
                feed_dict=x
            ),
            {self.input_var: data_x},
            out,
            batch_size
        )

        return out


def create_box_encoder(
    model_filename,
    input_name="images",
    output_name="features",
    batch_size=32
):

    image_encoder = ImageEncoder(
        model_filename,
        input_name,
        output_name
    )

    image_shape = image_encoder.image_shape

    def encoder(image, boxes):

        image_patches = []

        for box in boxes:

            patch = extract_image_patch(
                image,
                box,
                image_shape[:2]
            )

            if patch is None:

                patch = np.random.uniform(
                    0.0,
                    255.0,
                    image_shape
                ).astype(np.uint8)

            image_patches.append(patch)

        image_patches = np.asarray(image_patches)

        return image_encoder(
            image_patches,
            batch_size
        )

    return encoder


def generate_detections(
    encoder,
    mot_dir,
    output_dir,
    detection_dir=None
):

    if detection_dir is None:
        detection_dir = mot_dir

    try:
        os.makedirs(output_dir)

    except OSError as exception:

        if (
            exception.errno == errno.EEXIST
            and os.path.isdir(output_dir)
        ):
            pass

        else:
            raise ValueError(
                "Failed to create output directory"
            )

    for sequence in os.listdir(mot_dir):

        print("Processing", sequence)

        sequence_dir = os.path.join(
            mot_dir,
            sequence
        )

        image_dir = os.path.join(
            sequence_dir,
            "img1"
        )

        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
        }

        detection_file = os.path.join(
            detection_dir,
            sequence,
            "det/det.txt"
        )

        detections_in = np.loadtxt(
            detection_file,
            delimiter=","
        )

        detections_out = []

        frame_indices = detections_in[:, 0].astype(int)

        min_frame_idx = frame_indices.min()

        max_frame_idx = frame_indices.max()

        for frame_idx in range(
            min_frame_idx,
            max_frame_idx + 1
        ):

            print(
                "Frame %05d/%05d"
                % (frame_idx, max_frame_idx)
            )

            mask = frame_indices == frame_idx

            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                continue

            bgr_image = cv2.imread(
                image_filenames[frame_idx],
                cv2.IMREAD_COLOR
            )

            features = encoder(
                bgr_image,
                rows[:, 2:6].copy()
            )

            detections_out += [
                np.r_[row, feature]
                for row, feature in zip(rows, features)
            ]

        output_filename = os.path.join(
            output_dir,
            "%s.npy" % sequence
        )

        np.save(
            output_filename,
            np.asarray(detections_out),
            allow_pickle=False
        )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Re-ID Feature Extractor"
    )

    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to protobuf model"
    )

    parser.add_argument(
        "--mot_dir",
        required=True,
        help="Path to MOTChallenge directory"
    )

    parser.add_argument(
        "--detection_dir",
        default=None,
        help="Path to detection directory"
    )

    parser.add_argument(
        "--output_dir",
        default="detections",
        help="Output directory"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    encoder = create_box_encoder(
        args.model,
        batch_size=32
    )

    generate_detections(
        encoder,
        args.mot_dir,
        args.output_dir,
        args.detection_dir
    )


if __name__ == "__main__":
    main()