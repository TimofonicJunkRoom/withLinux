/*

reference : caffe code examples/mnist/convert_mnist_dataset.cpp

 */
#include <stdint.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

uint32_t
swap_endian (uint32_t val)
{
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0x00FF00FF);
  return (val << 16) | (val >> 16);
}

int
main (void)
{
  using namespace std;
  /* t10k-images-idx3-ubyte
   * t10k-labels-idx1-ubyte
   * train-images-idx3-ubyte
   * train-labels-idx1-ubyte */
  std::ifstream image_file("./t10k-images-idx3-ubyte",
    std::ios::in | std::ios::binary);
  std::ifstream label_file("./t10k-labels-idx1-ubyte",
    std::ios::in | std::ios::binary);

  uint32_t magic;
  uint32_t num_images;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  { // Header check

    image_file.read (reinterpret_cast<char *>(&magic), 4);
    magic = swap_endian(magic);
    assert(magic == 2051);
    cout << "Image file magic OK" << endl;

    label_file.read (reinterpret_cast<char *>(&magic), 4);
    magic = swap_endian(magic);
    assert(magic == 2049);
    cout << "Label file magic OK" << endl;

    image_file.read (reinterpret_cast<char *>(&num_images), 4);
    num_images = swap_endian(num_images);
    label_file.read (reinterpret_cast<char *>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    assert(num_images == num_labels);
    cout << num_images << " pairs of Image and Label" << endl;

    image_file.read (reinterpret_cast<char *>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read (reinterpret_cast<char *>(&cols), 4);
    cols = swap_endian(cols);
    cout << rows << " rows and " << cols << " cols each image." << endl;

  } // Header OK
  { // Dump the first image

    uint8_t * image = new uint8_t [rows * cols];
    image_file.read (reinterpret_cast<char *>(image), rows * cols);
    uint8_t * label = new uint8_t;
    label_file.read (reinterpret_cast<char *>(label), 1);

    cout << "Dump the first image" << endl;
    for (unsigned int i = 0; i < rows*cols; i++) {
      printf("%d ", image[i]);
    }
    printf("Label %d\n", *label);

  } // Dump the first image OK

  return 0;
}
