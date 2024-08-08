import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Mnist {
    public static class Data {
        public double[] train_images;
        public byte[] train_labels;
        public double[] test_images;
        public byte[] test_labels;

        public void deinit() {
            train_images = null;
            train_labels = null;
            test_images = null;
            test_labels = null;
        }
    }

    public static Data readMnist() throws IOException {
        Data data = new Data();

        String train_images_path = "data/train-images.idx3-ubyte";
        byte[] train_images_u8 = readIdxFile(train_images_path, 16);
        data.train_images = new double[784 * 60000];
        for (int i = 0; i < 784 * 60000; i++) {
            data.train_images[i] = (train_images_u8[i] & 0xFF) / 255.0;
        }

        String train_labels_path = "data/train-labels.idx1-ubyte";
        data.train_labels = readIdxFile(train_labels_path, 8);

        String test_images_path = "data/t10k-images.idx3-ubyte";
        byte[] test_images_u8 = readIdxFile(test_images_path, 16);
        data.test_images = new double[784 * 10000];
        for (int i = 0; i < 784 * 10000; i++) {
            data.test_images[i] = (test_images_u8[i] & 0xFF) / 255.0;
        }

        String test_labels_path = "data/t10k-labels.idx1-ubyte";
        data.test_labels = readIdxFile(test_labels_path, 8);

        return data;
    }

    public static byte[] readIdxFile(String path, int skipBytes) throws IOException {
        Path filePath = Paths.get(path);
        byte[] fileBytes = Files.readAllBytes(filePath);
        return Arrays.copyOfRange(fileBytes, skipBytes, fileBytes.length);
    }
}
