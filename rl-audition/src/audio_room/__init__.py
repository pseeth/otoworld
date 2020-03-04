from gym.envs.registration import register

register(
    id="audio-room-v0", entry_point="audio_room.envs:AudioEnv",
)
