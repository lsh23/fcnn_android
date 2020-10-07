package com.deephearing.se_android;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.PorterDuff;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.deephearing.se_android.gather.AudioImpl;
import com.deephearing.se_android.gather.IAudioCallback;
import com.deephearing.se_android.player.AudioPlayer;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

import io.agora.rtc.Constants;
import io.agora.rtc.IAudioFrameObserver;
import io.agora.rtc.IRtcEngineEventHandler;
import io.agora.rtc.RtcEngine;

public class ChatRoomActivity extends AppCompatActivity implements IAudioCallback, IAudioFrameObserver {

    // Constants that control the behavior of the speech enhancement code and model
    private static final int INPUT_DATA_LENGTH = 128;
    private static final int OUTPUT_DATA_LENGTH = 128;
    private static final String MODEL_FILENAME = "file:///android_asset/converted_model.tflite";

    private final static String TAG = ChatRoomActivity.class.getSimpleName();
    private TextView mTvInfoDisplay;

    private String mStrChannelName;
    private AudioEnum mAE = AudioEnum.App2SDK;
    private AudioProfile mAP;
    private ChannelProfile mCP;
    private int mChannleProfile;

    private RtcEngine mRtcEngine = null;
    private int samplingRate; // debug, use the fixed value
    private AudioPlayer mAudioPlayer = null;
    private AudioImpl mAI = null;
    private static final double sampleInterval = 0.01; //  sampleInterval >= 0.01
    private int channels = 2; // 1: Mono, 2: Stereo
    private int samplesPerCall = 0;

    private Interpreter tfLite;
    private boolean isActivated = false;

    // input tensor
    float[][] input = new float[1][128];
    float[][] wav = new float[1][128];
    float[][][][] spec = new float[1][2][129][1];
    float[][][][] inx10 = new float[1][16][2][56];
    float[][][][] inx12 = new float[1][32][2][56];
    float[][][][] inx14 = new float[1][64][2][56];
    float[][][][] inx16 = new float[1][128][2][28];
    float[][][][] inx2 = new float[1][64][2][14];
    float[][][][] inx4 = new float[1][32][2][28];
    float[][][][] inx6 = new float[1][16][2][28];
    float[][][][] inx8 = new float[1][8][2][28];
    float[][][][] inx = new float[1][128][2][2];

    //output tensor
    float[][][][] x10 = new float[1][16][1][56];
    float[][][][] x12 = new float[1][32][1][56];
    float[][][][] x14 = new float[1][64][1][56];
    float[][][][] x16 = new float[1][128][1][28];
    float[][][][] x2 = new float[1][64][1][14];
    float[][][][] x4 = new float[1][32][1][28];
    float[][][][] x6 = new float[1][16][1][28];
    float[][][][] x8 = new float[1][8][1][28];
    float[][][][] x = new float[1][128][1][2];
    float[] output = new float[128];


    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
        throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    IRtcEngineEventHandler mEngineHandler = new IRtcEngineEventHandler() {
        @Override
        public void onJoinChannelSuccess(String channel, int uid, int elapsed) {
            super.onJoinChannelSuccess(channel, uid, elapsed);
            sendMessage("onJoinChannelSuccess:" + (uid & 0xFFFFFFFFL));

            if (mAE == AudioEnum.App2App || mAE == AudioEnum.App2SDK) {
                mAI.start();
            }
        }

        @Override
        public void onRejoinChannelSuccess(String channel, int uid, int elapsed) {
            super.onRejoinChannelSuccess(channel, uid, elapsed);
        }

        @Override
        public void onError(int err) {
            super.onError(err);
        }

        @Override
        public void onApiCallExecuted(int error, String api, String result) {
            super.onApiCallExecuted(error, api, result);
            sendMessage("ApiCallExecuted:" + api);
        }

        @Override
        public void onLeaveChannel(IRtcEngineEventHandler.RtcStats stats) {
            Log.e(TAG, "onLeaveChannel");
            super.onLeaveChannel(stats);
        }

        @Override
        public void onUserJoined(int uid, int elapsed) {
            super.onUserJoined(uid, elapsed);
            sendMessage("onUserJoined:" + (uid & 0xFFFFFFFFL));
        }

        @Override
        public void onUserOffline(int uid, int reason) {
            super.onUserOffline(uid, reason);
            sendMessage("onUserOffLine:" + (uid & 0xFFFFFFFFL));
        }

        @Override
        public void onUserMuteAudio(int uid, boolean muted) {
            super.onUserMuteAudio(uid, muted);
            sendMessage("onUserMuteAudio:" + (uid & 0xFFFFFFFFL));
        }

        @Override
        public void onConnectionLost() {
            super.onConnectionLost();
            sendMessage("onConnectionLost");
        }

        @Override
        public void onConnectionInterrupted() {
            super.onConnectionInterrupted();
            sendMessage("onConnectionInterrupted");
        }

        @Override
        public void onConnectionBanned() {
            super.onConnectionBanned();
        }

        @Override
        public void onAudioRouteChanged(int routing) {
            super.onAudioRouteChanged(routing);
        }

        @Override
        public void onFirstLocalAudioFrame(int elapsed) {
            super.onFirstLocalAudioFrame(elapsed);
            sendMessage("onFirstLocalAudioFrame:" + elapsed);
        }

        @Override
        public void onFirstRemoteAudioFrame(int uid, int elapsed) {
            super.onFirstRemoteAudioFrame(uid, elapsed);
            sendMessage("onFirstRemoteAudioFrame:" + elapsed);
        }

        @Override
        public void onRtcStats(IRtcEngineEventHandler.RtcStats stats) {
            super.onRtcStats(stats);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chat_room);

        // Load speech enhancement model file.
        // Input Tensor shape : (1, 512)
        // Output Tensor shape : (1, 512)
        String actualModelFilename = MODEL_FILENAME.split("file:///android_asset/", -1)[1];
        try {
            tfLite = new Interpreter(loadModelFile(getAssets(), actualModelFilename));
        } catch (Exception e) {
            throw new RuntimeException();
        }
//        for (int i = 0; i < tfLite.getInputTensorCount(); i++) {
//            Tensor tensor = tfLite.getInputTensor(i);
//        }
//
//        for (int i = 0; i < tfLite.getOutputTensorCount(); i++) {
//            Tensor tensor2 =tfLite.getOutputTensor(i);
//        }
        tfLite.resizeInput(0, new int[] {INPUT_DATA_LENGTH, 1});

        initWidget();

        initAction();
        initAgoraEngine();
        dispatchWork();

        joinChannel();

        if (mAE == AudioEnum.App2App || mAE == AudioEnum.SDK2App) {
            mAudioPlayer.startPlayer();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private void initAction() {
        Intent mIntent = getIntent();
        mStrChannelName = mIntent.getStringExtra(IOConstants.CHANNEL_NAME);
//        mAE = (AudioEnum) mIntent.getSerializableExtra(IOConstants.AUDIO_ENUM);
        mAE = AudioEnum.App2SDK;
//        mAP = (AudioProfile) mIntent.getSerializableExtra(IOConstants.AUDIO_PROFILE);
        mAP = AudioProfile.AUDIO_PROFILE_16000;
//        mCP = (ChannelProfile) mIntent.getSerializableExtra(IOConstants.CHANNEL_PROFILE);
        mCP = ChannelProfile.CHANNEL_PROFILE_COMM;

        switch (mAP) {
            case AUDIO_PROFILE_8000:
                samplingRate = 8000;
                break;
            case AUDIO_PROFILE_16000:
                samplingRate = 16000;
                break;
            case AUDIO_PROFILE_32000:
                samplingRate = 32000;
                break;
            case AUDIO_PROFILE_44100:
                samplingRate = 44100;
            default:
                break;
        }

        switch (mCP) {
            case CHANNEL_PROFILE_COMM:
                mChannleProfile = Constants.CHANNEL_PROFILE_COMMUNICATION;
                break;
            case CHANNEL_PROFILE_LIVE:
                mChannleProfile = Constants.CHANNEL_PROFILE_LIVE_BROADCASTING;
                break;
            default:
                break;
        }


        mTvInfoDisplay.append("chose channel profile:" + mChannleProfile + "\n");
    }

    private void initWidget() {
        TextView mTvChannelName = findViewById(R.id.tv_channel_room);
        mTvInfoDisplay = findViewById(R.id.tv_info_display);

        mTvChannelName.setText(mStrChannelName);
    }

    public void onMuteClick(View v) {
        ImageView vi = (ImageView) v;
        if (mRtcEngine != null) {
            if (v.getTag() == null) {
                v.setTag(false);
            }
            boolean b = ((boolean) v.getTag());
            if (!b) {
                vi.setColorFilter(getResources().getColor(R.color.agora_blue), PorterDuff.Mode.MULTIPLY);
                mRtcEngine.muteLocalAudioStream(true);
            } else {
                vi.clearColorFilter();
                mRtcEngine.muteLocalAudioStream(false);
            }
            v.setTag(!b);

        }
    }

    public void onHungUpClick(View v) {
        dispatchFinish();
    }

    public void onEarPhone(View v) {
        ImageView vi = (ImageView) v;
        if (mRtcEngine != null) {
            if (v.getTag() == null) {
                v.setTag(true);
            }
            boolean b = ((boolean) v.getTag());
            if (b) {
                vi.setColorFilter(getResources().getColor(R.color.agora_blue), PorterDuff.Mode.MULTIPLY);
                mRtcEngine.setEnableSpeakerphone(true);
            } else {
                vi.clearColorFilter();
                mRtcEngine.setEnableSpeakerphone(false);
            }
            v.setTag(!b);
        }
    }

    public void onInferenceClick(View v) {
        ImageView vi = (ImageView) v;
        if (tfLite != null) {
            if (v.getTag() == null) {
                v.setTag(true);
            }
            boolean b = ((boolean) v.getTag());
            if (b) {
                vi.setColorFilter(getResources().getColor(R.color.agora_blue), PorterDuff.Mode.MULTIPLY);
                isActivated = true;
            } else {
                vi.clearColorFilter();
                isActivated = false;
            }
            v.setTag(!b);
        }
    }

    private void initAgoraEngine() {
        try {
            if (mRtcEngine == null) {
                Log.d(TAG, "== initAgoraEngine ==");
                mRtcEngine = RtcEngine.create(getBaseContext(), getString(R.string.app_key), mEngineHandler);

                mRtcEngine.setChannelProfile(mChannleProfile);
                if (mChannleProfile == Constants.CHANNEL_PROFILE_LIVE_BROADCASTING) {
                    mRtcEngine.setClientRole(Constants.CLIENT_ROLE_BROADCASTER);
                }

                mRtcEngine.setEnableSpeakerphone(false);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onAudioDataAvailable(long timeStamp, float[] audioData) {
        int size = audioData.length;
        byte[] data = new byte[size*2];
        float[][] floatInputBuffer = new float[1][INPUT_DATA_LENGTH];
        float[][] floatOutputBuffer = new float[1][OUTPUT_DATA_LENGTH];

        //float[][]
        short audioSample;

        // reshape (3200) to (3200, 1)
        for (int i = 0; i < INPUT_DATA_LENGTH; i++) {
            floatInputBuffer[0][i] = audioData[i % 128];
        }

        Object[] inputArray = {floatInputBuffer};
        Object[] outputArray = {floatOutputBuffer};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, floatOutputBuffer);

        //long currentThreadTimeMillisStart = System.nanoTime();
        //long currentThreadTimeMillisEnd = System.nanoTime();
        // Run the model.
        if (isActivated) {
            tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
//            tfLite.run(inputArray, outputArray);
        }
        //double elapsedSeconds = (currentThreadTimeMillisEnd - currentThreadTimeMillisStart)/1000000;


        //Log.d(TAG, "onAudioDataAvailable: "+ elapsedSeconds);
        for (int i = 0; i < size; i++) {
            if (isActivated) {
                audioSample = (short) ((floatOutputBuffer[0][i] * 32767.5) - 0.5);
            } else {
                audioSample = (short) ((audioData[i] * 32767.5) - 0.5);
            }
            data[2 * i] = (byte) (audioSample & 0x00FF);
            data[(2 * i) + 1] = (byte) ((audioSample & 0xFF00) >> 8);
//            short value = (short)(8000 * Math.sin(2. * Math.PI * 480.0 * (i/16000.0)));
        }
        mRtcEngine.pushExternalAudioFrame(data, timeStamp);
    }

    @Override
    public boolean onRecordFrame(byte[] bytes, int i, int i1, int i2, int i3) {
        Log.e(TAG, "=== onRecordFrame ====");
        return true;
    }

    @Override
    public boolean onPlaybackFrame(final byte[] bytes, int i, int i1, int i2, final int i3) {
        if (mAudioPlayer != null) {
            mAudioPlayer.play(bytes, 0, bytes.length);
        }

        return true;
    }

    private void dispatchWork() {
        //The algorithms for samplesPerCall of setPlaybackAudioFrameParameters()
        samplesPerCall = (int) (samplingRate * channels * sampleInterval);
        Log.e(TAG, "App numOfSamples: " + samplesPerCall);
        switch (mAE) {
            case App2App:
                doApp2App();
                break;
            case App2SDK:
                doApp2Sdk();
                break;
            case SDK2App:
                doSdk2App();
                break;
            case SDK2SDK:
                doSdk2Sdk();
                break;
            default:
                Log.e(TAG, "error on dispatchWork!");
                break;
        }
    }

    private void dispatchFinish() {
        switch (mAE) {
            case App2App:
                finishApp2App();
                break;
            case App2SDK:
                finishApp2Sdk();
                break;
            case SDK2App:
                finishSdk2App();
                break;
            case SDK2SDK:
                finishSdk2Sdk();
                break;
            default:
                Log.e(TAG, "error on dispatchFinish!");
                break;
        }
        leaveChannel();
        finish();
    }

    private void doApp2App() {
        mTvInfoDisplay.append("enter App2App mode!\n");

        startAudioGather(samplingRate, channels);
        startAudioPlayer(AudioManager.STREAM_VOICE_CALL, samplingRate, channels, AudioFormat.ENCODING_PCM_16BIT);

        mRtcEngine.setExternalAudioSource(true, samplingRate, channels);
        mRtcEngine.setParameters("{\"che.audio.external_render\": true}");
        mRtcEngine.registerAudioFrameObserver(this);
        mRtcEngine.setPlaybackAudioFrameParameters(samplingRate, channels, 0, samplesPerCall);
    }

    private void finishApp2App() {
        mRtcEngine.registerAudioFrameObserver(null);
        mRtcEngine.setParameters("{\"che.audio.external_render\": false}");
        mRtcEngine.setExternalAudioSource(false, samplingRate, channels);
        finishAudioGather();
        finishAudioPlayer();
    }

    private void doApp2Sdk() {
        startAudioGather(samplingRate, channels);
        mRtcEngine.setExternalAudioSource(true, samplingRate, channels);
        mRtcEngine.setParameters("{\"che.audio.external_render\": false}");
        mTvInfoDisplay.append("enter App2SDK mode!\n");
    }

    private void finishApp2Sdk() {
        mRtcEngine.setExternalAudioSource(false, samplingRate, channels);
        finishAudioGather();
    }

    private void doSdk2App() {
        startAudioPlayer(AudioManager.STREAM_VOICE_CALL, samplingRate, channels, AudioFormat.ENCODING_PCM_16BIT);
        mRtcEngine.setPlaybackAudioFrameParameters(samplingRate, channels, 0, samplesPerCall);
        mRtcEngine.setParameters("{\"che.audio.external_render\": true}");
        mTvInfoDisplay.append("enter SDK2App mode!\n");
        mRtcEngine.registerAudioFrameObserver(this);
    }

    private void finishSdk2App() {
        finishAudioPlayer();
        mRtcEngine.registerAudioFrameObserver(null);
        mRtcEngine.setParameters("{\"che.audio.external_render\": false}");
    }

    private void doSdk2Sdk() {
        mTvInfoDisplay.append("enter SDK2SDK mode!\n");
    }

    private void finishSdk2Sdk() {
    }

    private void startAudioGather(int samplingRate, int channelConfig) {
        if (mAI == null) {
            mAI = new AudioImpl(samplingRate, channelConfig);
        }
        mAI.init(this);
    }

    private void finishAudioGather() {
        if (mAI != null) {
            mAI.stop();
            mAI.destroy();
        }
    }

    private void startAudioPlayer(int streamType, int sampleRateInHz, int channelConfig, int audioFormat) {
        if (mAudioPlayer == null) {
            mAudioPlayer = new AudioPlayer(streamType, sampleRateInHz, channelConfig, audioFormat);
        }
    }

    private void finishAudioPlayer() {
        if (mAudioPlayer != null) {
            mAudioPlayer.stopPlayer();
        }
    }

    private void joinChannel() {
        int ret = mRtcEngine.joinChannel(null, mStrChannelName.trim(), getResources().getString(R.string.app_key), 0);
        if (null != mRtcEngine)
            Log.e(TAG, "SDK Ver: " + mRtcEngine.getSdkVersion() + " ret : " + ret);
    }

    private void leaveChannel() {
        mRtcEngine.leaveChannel();
    }

    private void sendMessage(@NonNull final String s) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mTvInfoDisplay.append(s + "\n");
            }
        });
    }
}