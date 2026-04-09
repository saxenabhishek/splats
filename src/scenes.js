// Curated scene list for benchmarking.
// Splat count is estimated from file size: 1 splat ≈ 32 bytes in .splat format.

const CAKEWALK = "https://huggingface.co/cakewalk/splat-data/resolve/main/";
const VLAD = "https://huggingface.co/VladKobranov/splats/resolve/main/";

export const scenes = [
  // These are the 3DGS benchmark scenes, good for apples-to-apples comparison.
  {
    label: "Nike (8.7 MB 285K splats)",
    url: CAKEWALK + "nike.splat",
    group: "cakewalk",
  },
  {
    label: "Plush (9 MB 295K splats)",
    url: CAKEWALK + "plush.splat",
    group: "cakewalk",
  },
  {
    label: "Train (32.8 MB 1.1M splats)",
    url: CAKEWALK + "train.splat",
    group: "cakewalk ",
  },
  {
    label: "Room (51 MB 1.7M splats)",
    url: CAKEWALK + "room.splat",
    group: "cakewalk ",
  },
  {
    label: "Truck (81.3 MB 2.7M splats)",
    url: CAKEWALK + "truck.splat",
    group: "cakewalk ",
  },
  {
    label: "Treehill (121 MB 4M splats)",
    url: CAKEWALK + "treehill.splat",
    group: "cakewalk ",
  },
  {
    label: "Stump (159 MB 5.2M splats)",
    url: CAKEWALK + "stump.splat",
    group: "cakewalk ",
  },
  {
    label: "Bicycle (196 MB 6.4M splats)",
    url: CAKEWALK + "bicycle.splat",
    group: "cakewalk ",
  },
  {
    label: "Garden (187 MB 6.1M splats)",
    url: CAKEWALK + "garden.splat",
    group: "cakewalk ",
  },

  // Good variety of sizes and subject matter.
  {
    label: "Baby Yoda (5.3 MB 175K splats)",
    url: VLAD + "baby_yoda.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Bonsai (7.5 MB 245K splats)",
    url: VLAD + "bonsai.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Flower (6.7 MB 220K splats)",
    url: VLAD + "flower.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Bicycle no-road (6.1 MB 200K splats)",
    url: VLAD + "bicycle-noroad.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Chair modern (6.1 MB 200K splats)",
    url: VLAD + "chair_gray_modern.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Car Toyota (9.5 MB 310K splats)",
    url: VLAD + "car-toyota-white.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Kotofuri (11.4 MB 375K splats)",
    url: VLAD + "2024march-kotofuri.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Actus (19.2 MB 630K splats)",
    url: VLAD + "actus.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Dialogue (24.5 MB 800K splats)",
    url: VLAD + "dialogue.splat",
    group: "VladKobranov · objects",
  },
  {
    label: "Fountain light (20.3 MB 665K splats)",
    url: VLAD + "fountain-light.splat",
    group: "VladKobranov · scenes",
  },
  {
    label: "Fountain full (34.4 MB 1.1M splats)",
    url: VLAD + "fountain-full.splat",
    group: "VladKobranov · scenes",
  },
  {
    label: "Kotofuri full (31.1 MB 1M splats)",
    url: VLAD + "2024march-kotofuri-full.splat",
    group: "VladKobranov · scenes",
  },
];
