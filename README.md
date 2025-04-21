# GANS-MODEL
datasets links
 https://www.kaggle.com/datasets/robgonsalves/impressionistlandscapespaintings
 https://www.kaggle.com/datasets/ashwingupta3012/human-faces


# Real to Landscape Painting Conversion using CycleGAN

This project uses a **Generative Adversarial Network (GAN)** — specifically **CycleGAN** — to transform **real landscape photographs** into **artistic landscape paintings**. The model learns to mimic the style of paintings while preserving the structure of real-world photos, making it ideal for creative AI applications.

---

## Project Description

CycleGAN is used for **unpaired image-to-image translation**. It learns the mapping between two domains without requiring corresponding image pairs.

- **Domain A:** Real-world landscape images (e.g., photos)
- **Domain B:** Artistic landscape paintings

This project demonstrates how CycleGAN can convert a real image into an artwork-style image, such as turning a photo into a painting inspired by famous styles (like Monet, Van Gogh, etc.).

---

##  Model Overview

CycleGAN consists of:
- Two Generators (A → B and B → A)
- Two Discriminators (for domain A and B)
- Uses **cycle-consistency loss** and **adversarial loss**

---

