import { makeLevelCamera } from "./math.js";

// Orbit cameras around the scene origin, all level on the horizon.
// work well for the cakewalk research scenes (stump, train, truck etc.)

const CENTER = [0, 0, 0];
const R = 4; // orbit radius (units from center)
const H = 1.5; // eye height above center

export const cameras = [
  makeLevelCamera([-R, H, 0], CENTER), // 0 — left
  makeLevelCamera([0, H, -R], CENTER), // 1 — front
  makeLevelCamera([R, H, 0], CENTER), // 2 — right
  makeLevelCamera([0, H, R], CENTER), // 3 — back

  // Diagonal positions
  makeLevelCamera([-R, H, -R], CENTER), // 4 — front-left
  makeLevelCamera([R, H, -R], CENTER), // 5 — front-right
  makeLevelCamera([R, H, R], CENTER), // 6 — back-right
  makeLevelCamera([-R, H, R], CENTER), // 7 — back-left

  // Elevated views
  makeLevelCamera([0, R, 0], CENTER), // 8 — top-down 
  makeLevelCamera([0, R / 2, -R * 1.5], CENTER), // 9 — raised front
];

// Default view: slightly elevated front-left
export const defaultViewMatrix = [
  0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07, 0.03,
  6.55, 1,
];
