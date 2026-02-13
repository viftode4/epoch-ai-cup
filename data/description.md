
Dataset Description
Radar tracks were generated using the MAX Avian Radar, while visual observations were collected by bird spotters. All data comes from Windpark Eemshaven in the province of Groningen.

Files
The dataset consists out of 2 files: train_data.csv and test_data.csv. Each CSV contains two types of information: radar-derived data and observation-related data. Both files include the following radar-related columns:

track_id — The unique identifier for each radar track
timestamp_start_radar_utc and timestamp_end_radar_utc — The start and end time of the radar track in UTC
trajectory — The actual radar track encoded in EWKB (Extended Well-Known Binary) Hex. Data consists out of a series of measurements containing 4 elements in order: Longitude, Latitude, Altitude (m), Radar Cross Section (dB/m2)
trajectory_time — A list of floating numbers indicating the elapsed time in seconds since timestamp_start_radar_utc a measurement in the trajectory was made
radar_bird_size — Which size of bird the radar track could belong to (Large, Medium, Small bird, Flock) as output from the MAX Avian Radar
airspeed — Average airspeed of the radar track (m/s)
min_z — Minimum altitude of the radar track compared to the height of the radar (m)
max_z — Maximum altitude of the radar track compared to the height of the radar (m)
Then, train_data.csv also includes observation-based columns used to validate the radar tracks. These are privileged columns, which are only available during training-time. The target labels for bird group classification are in the column bird_group . The following columns are included:

observation_id — The unique identifier of each observation
primary_observation_id — The unique identifier of each unique observation. (The same bird (flock) can consist of multiple radar tracks)
observer_position — The location of the observer encoded in EWKB (Extended Well-Known Binary) Hex as Longitude / Latitude / Altitude
observer_comment — A comment left by the observer regarding the observation
n_birds_observed — The amount of birds observed
bird_group — The bird group of the radar track
bird_species — The exact species of the radar track
Finally, a sample_submission.csv is provided. The sample submission has the following format:

track_id,Clutter,Cormorants,Pigeons,Ducks,Geese,Gulls,Birds of Prey,Waders,Songbirds
526781579,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
526782173,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
526782843,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
526786562,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
526787997,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
526788673,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
526788796,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
526788552,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
Here, each row corresponds to a track_id (the unique identifier for each radar track in the test set) with a score between 0-1 for the class Clutter and each of the bird groups.