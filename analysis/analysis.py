import numpy as np
import cv2
import os

class AnalysisPictures:
    def __init__(self, player_positions):
        self.player_positions = player_positions
        self.output_folder = "D:\\Football_Analysis_Detection\\output"
        self.accumulated_heatmap = np.zeros((720, 1280), dtype=np.float32)  # Initialize cumulative heatmap
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def generate_heatmap(self):
        # Load football pitch image
        pitch_img = cv2.imread('D:\\Football_Analysis_Detection\\football_track.jpg')
        pitch_img = cv2.resize(pitch_img, (1280, 720))  # Resize to match video dimensions

        print("Collecting player positions for heatmap:")

        for player_id, positions in self.player_positions.items():
            print(f"Player ID {player_id}: {positions}")  # Print each player's positions

            for pos in positions:
                # Check if position is within bounds
                if 0 <= int(pos[0]) < 1280 and 0 <= int(pos[1]) < 720:
                    cv2.circle(self.accumulated_heatmap, (int(pos[0]), int(pos[1])), 10, (1), -1)

        # Normalize the heatmap
        cv2.normalize(self.accumulated_heatmap, self.accumulated_heatmap, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.array(self.accumulated_heatmap, dtype=np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_AUTUMN)

        # Blend heatmap with pitch image
        heatmap_overlay = cv2.addWeighted(pitch_img, 0.6, heatmap_color, 0.4, 0)

        # Save heatmap overlay
        cv2.imwrite(os.path.join(self.output_folder, 'heatmap_overlay.png'), heatmap_overlay)

    def generate_zone_coverage(self, grid_size=(5, 5), frame_size=(1280, 720)):
        """
        Generate zone coverage analysis.
        Divide the pitch into zones and count player presence.
        """
        # Load football pitch image
        pitch_img = cv2.imread('D:\\Football_Analysis_Detection\\football_track.jpg')
        pitch_img = cv2.resize(pitch_img, frame_size)  # Resize to the correct frame size

        # Create a grid to count player coverage in each zone (not overwriting the pitch image)
        coverage_grid = np.zeros(grid_size, dtype=np.int32)

        # Calculate player presence in each zone
        for positions in self.player_positions.values():
            for pos in positions:
                x, y = pos
                zone_x = int(x // (frame_size[0] / grid_size[0]))
                zone_y = int(y // (frame_size[1] / grid_size[1]))
                coverage_grid[min(zone_x, grid_size[0] - 1), min(zone_y, grid_size[1] - 1)] += 1

        # Normalize the coverage grid for visualization (convert to [0, 255] range)
        normalized_coverage = cv2.normalize(coverage_grid, None, 0, 255, cv2.NORM_MINMAX)

        # Create a heatmap-like visual for zone coverage
        zone_coverage_img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        zone_height = frame_size[1] // grid_size[1]
        zone_width = frame_size[0] // grid_size[0]

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                color_intensity = normalized_coverage[i, j]
                cv2.rectangle(zone_coverage_img, (i * zone_width, j * zone_height),
                            ((i + 1) * zone_width, (j + 1) * zone_height),
                            (0, int(color_intensity), 255 - int(color_intensity)), -1)

        # Blend the zone coverage with the pitch image (using addWeighted for transparency)
        blended_img = cv2.addWeighted(pitch_img, 0.6, zone_coverage_img, 0.4, 0)

        # Resize the zone coverage image to a smaller size for display
        zone_coverage_img_small = cv2.resize(blended_img, (300, 200), interpolation=cv2.INTER_AREA)

        # Save the blended image for use in main.py
        cv2.imwrite(os.path.join(self.output_folder, 'zone_coverage.png'), zone_coverage_img_small)

        return zone_coverage_img_small