package com.deephearing.se_android.gather;

public interface IAudioCallback {
    void onAudioDataAvailable(long timeStamp, float[] audioData);
}
