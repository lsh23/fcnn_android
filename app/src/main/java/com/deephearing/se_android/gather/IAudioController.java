package com.deephearing.se_android.gather;


public interface IAudioController {
    AudioStatus init(IAudioCallback callback);
    AudioStatus start();
    AudioStatus stop();
    void destroy();
}
