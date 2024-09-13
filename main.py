from utils import read_video, save_video
from trackers import Tracker
import cv2
import os
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from analysis import AnalysisPictures

def main():
    # Read Video
    video_frames = read_video('D:\\Football_Analysis_Detection\\video\\input_video\\video.mp4')

    # Initialize Tracker
    tracker = Tracker('D:\\Football_Analysis_Detection\\models\\best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        if not player_track:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None)
            continue

        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                last_player = team_ball_control[-1]
                team_ball_control.append(last_player)

    team_ball_control = np.array(team_ball_control)

    # Collect player positions
    player_positions = {}
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            if isinstance(track, dict) and 'bbox' in track and isinstance(track['bbox'], (list, np.ndarray)):
                player_positions.setdefault(player_id, []).append(track['bbox'][:2])

    # Initialize AnalysisPictures with collected data
    analysis_pictures = AnalysisPictures(player_positions)

    # Parameters for video overlay timing
    fps = 30  # Assuming 30 FPS video
    heatmap_duration = 12 * fps  # Heatmap for first 10 seconds
    zone_coverage_duration = 12 * fps  # Zone coverage for next 10 seconds
    window_size = 30  # Number of frames for the heatmap window

    # Main loop for processing video frames
    output_video_frames = []
    for frame_num in range(len(video_frames)):
        recent_positions = {player_id: positions[max(0, frame_num - window_size):frame_num]
                            for player_id, positions in player_positions.items()}

        # Update AnalysisPictures with recent positions
        analysis_pictures = AnalysisPictures(recent_positions)

        # Overlay heatmap
        if frame_num < heatmap_duration:  # First 10 seconds: display heatmap
            analysis_pictures.generate_heatmap()  # Generate updated heatmap
            heatmap_overlay = cv2.imread(os.path.join(analysis_pictures.output_folder, 'heatmap_overlay.png'))

            # Resize heatmap to smaller size (e.g., 300x200)
            heatmap_small = cv2.resize(heatmap_overlay, (300, 200), interpolation=cv2.INTER_AREA)

            # Overlay heatmap in the defined box (move to bottom-left corner)
            frame = video_frames[frame_num]
            if heatmap_small is not None and frame is not None:
                frame[frame.shape[0] - 200:frame.shape[0], 20:320] = heatmap_small  # Move to bottom-left corner
                cv2.putText(frame, "Heatmap", (20, frame.shape[0] - 210), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)

        # Overlay zone coverage
        elif heatmap_duration <= frame_num < heatmap_duration + zone_coverage_duration: 
            zone_coverage_img = analysis_pictures.generate_zone_coverage()  # Generate zone coverage image

            # Resize zone coverage image to smaller size (e.g., 300x200)
            zone_coverage_img_resized = cv2.resize(zone_coverage_img, (300, 200), interpolation=cv2.INTER_AREA)

            # Overlay zone coverage in the defined box (move to bottom-left corner)
            frame = video_frames[frame_num]
            if zone_coverage_img_resized is not None and frame is not None:
                frame[frame.shape[0] - 200:frame.shape[0], 20:320] = zone_coverage_img_resized  # Move to bottom-left corner
                cv2.putText(frame, "Zone Coverage", (20, frame.shape[0] - 210), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)

        # If needed, add any other analysis or simply proceed without heatmap dots
        elif frame_num >= heatmap_duration + zone_coverage_duration:
            frame = video_frames[frame_num]  # Just keep the frame as it is or apply other logic if needed

        output_video_frames.append(frame)

    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save output video with overlays
    save_video(output_video_frames, 'D:\\Football_Analysis_Detection\\video\\output_videos\\video.avi')

if __name__ == '__main__':
    main()
