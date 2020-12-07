package com.deephearing.se_android;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Color;
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
    private static final int INPUT_TENSOR_SIZE = 1152;
    private static final int INPUT_BUFFER_SIZE = 896;
    private static final int OUTPUT_DATA_LENGTH = 896;
    private static final String MODEL_FILENAME = "file:///android_asset/model.tflite";

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
    private int channels = 1; // 1: Mono, 2: Stereo
    private int samplesPerCall = 0;

    private Interpreter tfLite;
    private final Interpreter.Options tfLiteOptions = new Interpreter.Options();
    private boolean isActivated = false;

    float[][] floatInputBuffer = new float[1][INPUT_TENSOR_SIZE];

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
        // Input Tensor shape : (1, 1152) 1152 = [128, 896(output), 128]
        // Output Tensor shape : (1, 896)
        String actualModelFilename = MODEL_FILENAME.split("file:///android_asset/", -1)[1];
        try {
//            tfLite = new Interpreter(loadModelFile(getAssets(), actualModelFilename));
            tfLite = new Interpreter(loadModelFile(getAssets(), actualModelFilename), tfLiteOptions);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            throw new RuntimeException();
        }
        for (int i = 0; i < tfLite.getInputTensorCount(); i++) {
            Tensor tensor = tfLite.getInputTensor(i);
        }

        for (int i = 0; i < tfLite.getOutputTensorCount(); i++) {
            Tensor tensor2 =tfLite.getOutputTensor(i);
        }
//        tfLite.resizeInput(0, new int[] {1, INPUT_TENSOR_SIZE});

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
        mAE = AudioEnum.App2SDK;
        mAP = AudioProfile.AUDIO_PROFILE_8000;
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
                vi.setColorFilter(getResources().getColor(R.color.colorPrimary), PorterDuff.Mode.MULTIPLY);
                mRtcEngine.muteLocalAudioStream(true);
            } else {
                vi.setColorFilter(Color.parseColor("#B5B3DC"), PorterDuff.Mode.MULTIPLY);
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
                vi.setColorFilter(getResources().getColor(R.color.colorPrimary), PorterDuff.Mode.MULTIPLY);
                mRtcEngine.setEnableSpeakerphone(true);
            } else {
                vi.setColorFilter(Color.parseColor("#B5B3DC"), PorterDuff.Mode.MULTIPLY);
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
                TextView textView = (TextView) findViewById(R.id.textView2);
                textView.setText("ON\nSpeech Enhancement");
                vi.setColorFilter(getResources().getColor(R.color.agora_blue), PorterDuff.Mode.MULTIPLY);
                isActivated = true;
            } else {
                TextView textView = (TextView) findViewById(R.id.textView2);
                textView.setText("OFF\nSpeech Enhancement");
                vi.setColorFilter(Color.parseColor("#DDDEEC"), PorterDuff.Mode.MULTIPLY);
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
    double sum = 0;
    int count = 1;
    @Override
    public void onAudioDataAvailable(long timeStamp, float[] audioData) {
        float [][] test_input = {{-0.023112802F,-0.026694486F,-0.0248926F,-0.02201704F,-0.004586883F,0.0008148784F,-0.0075156116F,-0.007836137F,-0.0014399207F,0.0023003272F,0.021736871F,0.048615225F,0.05949516F,0.070282236F,0.05093868F,-0.0017857468F,-0.037668075F,-0.08274349F,-0.08334698F,-0.034292806F,0.00423017F,0.046290398F,0.054252606F,0.03245375F,0.015354328F,-0.002669121F,0.008690648F,0.028411096F,0.023257494F,0.021024877F,0.0036798893F,-0.018639604F,-0.01953798F,-0.044916213F,-0.061347496F,-0.054865297F,-0.044027008F,-0.003289674F,0.031690322F,0.043608624F,0.049176537F,0.014734255F,-0.020405225F,-0.015464693F,-0.00048331846F,0.010945121F,0.018002395F,0.0053693163F,-0.01928835F,-0.013802912F,-0.026202926F,-0.033331927F,-0.028014723F,-0.022185497F,0.0267034F,0.071569316F,0.08337339F,0.079123616F,0.027908394F,-0.018501014F,-0.029512947F,-0.05856438F,-0.044707906F,-0.015298416F,-0.011533685F,0.0022269282F,-0.0068337736F,-0.024282677F,-0.022292519F,-0.03777836F,-0.018490715F,0.023428721F,0.03589578F,0.05114992F,0.053535964F,0.022663392F,-0.008046548F,-0.058174238F,-0.10961412F,-0.09726197F,-0.044247277F,0.041552164F,0.112997435F,0.122838244F,0.081904195F,0.014724892F,-0.054528803F,-0.07048676F,-0.044453535F,0.0021648249F,0.064109795F,0.07962659F,0.053271323F,0.008136047F,-0.055399828F,-0.091007456F,-0.07948599F,-0.041764215F,0.036732644F,0.097818255F,0.12419955F,0.103404164F,0.024939455F,-0.041333012F,-0.08777974F,-0.112504445F,-0.088305555F,-0.043522507F,0.0020868257F,0.026330017F,0.011718806F,-0.0070673637F,-0.03558082F,-0.050967246F,-0.030208882F,0.0011293013F,0.024227936F,0.040944215F,0.055902265F,0.057206996F,0.031746823F,-0.0039835502F,-0.020756483F,-0.02313998F,-0.0033190784F,0.025395155F,0.01226299F,-0.0074906484F,-0.0076373527F,-0.0036322065F,0.010874769F,0.0046005156F,-0.0078233415F,0.006151667F,0.02674026F,0.049124036F,0.059114475F,0.024068225F,-0.015299169F,-0.034260184F,-0.04544617F,-0.03237657F,0.0037687225F,0.027106958F,0.03973745F,0.048307084F,0.0230827F,-0.025357377F,-0.08285554F,-0.10865024F,-0.09146369F,-0.059142146F,-0.005996464F,0.049218226F,0.06617512F,0.055378288F,0.0117351925F,-0.052989826F,-0.06499387F,-0.02979806F,0.006330196F,0.049621645F,0.07841133F,0.07059914F,0.05758722F,0.020707026F,-0.029886855F,-0.046145394F,-0.035006296F,0.0006011445F,0.029387103F,0.007467238F,-0.0083925845F,-0.018537736F,-0.022623101F,0.017870737F,0.05434888F,0.06729395F,0.051108584F,0.007907431F,-0.020608068F,-0.03659576F,-0.049745664F,-0.021349685F,0.0024888907F,0.008344501F,0.015079297F,-0.020429129F,-0.075255886F,-0.11892317F,-0.13598156F,-0.08777302F,-0.006935006F,0.05026808F,0.079913184F,0.07181914F,0.06334715F,0.083303966F,0.07351287F,0.0017302055F,-0.078077264F,-0.11538452F,-0.08479285F,-0.0068273023F,0.045159604F,0.051680647F,0.030537691F,0.008906276F,0.017278142F,0.015826955F,-0.006367065F,-0.0007114676F,0.013531984F,0.043289483F,0.06054519F,0.03442025F,0.00018138136F,-0.031923283F,-0.030584449F,-0.0052254833F,0.010645086F,-0.0031940993F,-0.02572906F,-0.032069054F,-0.024431108F,-0.011296887F,-0.007655654F,-0.018650813F,-0.03385062F,-0.042400952F,-0.023382332F,0.013417365F,0.025082879F,0.034989312F,-0.013540931F,-0.06957274F,-0.045745842F,0.0027572466F,0.059471324F,0.07509938F,0.028990751F,-0.001881225F,-0.001535187F,-0.0083494205F,0.0024352055F,0.0002755737F,0.0016764633F,0.03639756F,0.061346274F,0.056393042F,0.010941064F,-0.0447613F,-0.053942423F,-0.034714084F,-0.008789575F,0.026789628F,0.03503605F,0.042903367F,0.053739764F,0.055577338F,0.039652005F,-0.0027976297F,-0.04046632F,-0.04617409F,-0.04658788F,-0.04483819F,-0.030948043F,-0.037527468F,-0.052729554F,-0.062084265F,-0.04690499F,0.003938187F,0.048412543F,0.05768731F,0.04081384F,0.006905925F,0.0020307135F,0.029581307F,0.033462092F,-0.014325678F,-0.078755826F,-0.090728596F,-0.031641398F,0.051726848F,0.10782098F,0.10079296F,0.056886908F,0.025010904F,0.009973081F,-0.0056836903F,-0.04111084F,-0.057614144F,-0.029649656F,0.0028177393F,0.010929897F,0.012752403F,0.00013047544F,-0.0027115569F,0.016000928F,0.032030523F,0.035657175F,0.028789688F,0.030707149F,0.026330082F,0.009841357F,-0.024685085F,-0.07071605F,-0.095616564F,-0.09387719F,-0.099639215F,-0.081680566F,-0.024024826F,0.035106827F,0.0693146F,0.046185024F,0.011559758F,0.0035717152F,0.018989746F,0.05137968F,0.062649034F,0.018887516F,-0.022980079F,-0.016198784F,0.0130532365F,0.03370539F,-0.005448092F,-0.047210786F,-0.04576562F,-0.031116862F,0.012596375F,0.02826045F,0.0022840602F,0.010367321F,0.028670618F,0.054890092F,0.07680356F,0.035772838F,-0.011567487F,-0.028250255F,-0.014482504F,0.014400981F,0.00905997F,-0.016662277F,-0.0313078F,-0.05080246F,-0.053937756F,-0.042893462F,-0.062787205F,-0.07922824F,-0.08461144F,-0.05046397F,0.049290214F,0.12684035F,0.12980981F,0.06531655F,-0.032234054F,-0.06603651F,-0.0429046F,-0.027166668F,-0.0021750946F,-0.01220911F,-0.027165141F,0.021883737F,0.057359863F,0.051498182F,0.036196653F,0.0031043515F,0.005090809F,0.019561173F,-0.0074704383F,-0.00799034F,-0.012203712F,-0.010012392F,0.022305807F,0.009954432F,-0.0160338F,-0.036683824F,-0.029545896F,0.040030047F,0.08202883F,0.07955648F,0.05162403F,-0.005289887F,-0.02410413F,-0.043535646F,-0.06748374F,-0.06527045F,-0.07264281F,-0.07053229F,-0.06512566F,-0.058230806F,-0.009289399F,0.045215055F,0.09216027F,0.09634122F,0.0020710453F,-0.05990102F,-0.041957542F,0.0133426F,0.06462366F,0.041901052F,-0.024912518F,-0.05196321F,-0.006503038F,0.048398778F,0.04737618F,0.020320209F,0.0022440124F,0.0113987485F,0.04320348F,0.030685805F,-0.012322438F,-0.04164332F,-0.045067824F,-0.009170322F,0.026427988F,0.020053778F,0.0054616136F,0.010657191F,0.022975817F,0.0302244F,0.02412008F,0.0028001145F,-0.00306255F,0.011802718F,0.009081689F,0.0010964125F,-0.024535682F,-0.07293408F,-0.11022244F,-0.12843919F,-0.093448214F,-0.007069897F,0.076212175F,0.11748087F,0.058866467F,-0.038255394F,-0.06425099F,-0.023431584F,0.04670702F,0.067314155F,0.04575438F,0.03566658F,0.029318562F,0.043270815F,0.010597261F,-0.052855767F,-0.052792408F,-0.01923606F,0.03618528F,0.05145443F,-0.009865873F,-0.05656325F,-0.059847645F,-0.032166045F,0.025871832F,0.058120523F,0.07009756F,0.06668568F,0.04420972F,0.015968917F,-0.032976873F,-0.055313468F,-0.03785004F,-0.0031441832F,0.02434627F,0.011068704F,-0.019255808F,-0.045688443F,-0.09808629F,-0.12530862F,-0.09517336F,-0.01804232F,0.089023165F,0.13541862F,0.081719644F,-0.020544972F,-0.09312714F,-0.082361385F,-0.00834471F,0.044756297F,0.05034281F,0.058399547F,0.06434225F,0.049563043F,0.021048224F,-0.04867666F,-0.0818291F,-0.027053911F,0.03189183F,0.05445579F,0.03993884F,0.0057007372F,-0.008647829F,-0.005402156F,-0.013622617F,-0.02910626F,-0.04168217F,-0.021402951F,0.019168634F,0.054554716F,0.06943608F,0.053364564F,0.04162259F,0.024001855F,-0.011226604F,-0.050742395F,-0.07263591F,-0.08021605F,-0.07997767F,-0.08789092F,-0.10103225F,-0.05693423F,0.010515135F,0.070864394F,0.088420875F,0.031306274F,-0.015109373F,-0.002245076F,0.04584743F,0.07648121F,0.027916256F,-0.04318607F,-0.052811965F,-0.023799991F,0.019653555F,0.033169337F,0.02158728F,0.019336514F,0.013823134F,0.0069110524F,-0.035274826F,-0.077998996F,-0.05374284F,0.009196879F,0.07120919F,0.09154414F,0.06809121F,0.03164646F,0.001957735F,-0.016657937F,-0.033207737F,-0.039544966F,-0.016008707F,0.016818058F,0.032238F,0.020389363F,-0.010338565F,-0.025447205F,-0.0366105F,-0.067334756F,-0.121790685F,-0.13983758F,-0.07245071F,0.038251463F,0.11876129F,0.101912245F,0.0114805335F,-0.049363688F,-0.024561962F,0.04541345F,0.0736959F,0.040396996F,0.008571093F,0.0030738325F,0.021403845F,0.020224746F,-0.024820235F,-0.04294576F,-0.019321961F,0.012753793F,0.02850466F,0.0023943507F,-0.03052186F,-0.028311554F,0.0036821403F,0.029175393F,0.032157905F,0.045681298F,0.058417026F,0.063682005F,0.047978766F,-0.013838001F,-0.06628643F,-0.06502982F,-0.040160898F,-0.0060501895F,0.013212498F,0.005026024F,0.0014838576F,-0.024095584F,-0.073824525F,-0.1393821F,-0.13480473F,-0.022853494F,0.104885876F,0.16203037F,0.103210464F,0.00425373F,-0.034866445F,-0.014457117F,-0.014725931F,-0.041543037F,-0.06488415F,-0.017531741F,0.06382635F,0.10423234F,0.07649247F,-0.0054422263F,-0.024117999F,0.0027487306F,0.0036318041F,-0.015478491F,-0.033016898F,-0.031009143F,0.00825444F,0.03243547F,0.015482389F,-0.0015427743F,0.008142136F,0.0386289F,0.058299236F,0.047327995F,0.017148899F,-0.0013234941F,-0.01565794F,-0.039536987F,-0.061634477F,-0.06525397F,-0.041024707F,-0.007131012F,-0.007694591F,-0.05480006F,-0.07856646F,-0.020722581F,0.05305628F,0.07573295F,0.011999875F,-0.078880705F,-0.07791492F,0.0066888984F,0.084899F,0.097454906F,0.044458754F,-0.0024479795F,0.012816556F,0.014633305F,-0.016528232F,-0.045982704F,-0.032173812F,0.025973871F,0.04280079F,0.02735687F,-0.008976295F,-0.04403951F,-0.024399668F,-0.008463701F,-0.020296019F,-0.011260387F,0.01773043F,0.06508478F,0.09296529F,0.054528564F,0.015212299F,-0.0006813039F,-0.012420861F,-0.00800568F,-0.021504603F,-0.043501064F,-0.048891954F,-0.051609896F,-0.072717145F,-0.10015556F,-0.119354926F,-0.08160387F,0.03942822F,0.12618953F,0.123949565F,0.044629674F,-0.041704394F,-0.060129266F,-0.01930363F,0.024148561F,0.027733404F,0.018012162F,0.030961042F,0.06293813F,0.055236295F,-0.0013593724F,-0.06262313F,-0.058598742F,-0.002252806F,0.023950193F,0.014923461F,-0.009591397F,-0.017157437F,0.003492551F,0.006112068F,-0.0073301205F,-0.0037380243F,0.013094347F,0.043759055F,0.052972637F,0.0225415F,0.0013538375F,0.00027362024F,0.017609289F,0.02061979F,-0.0016862303F,-0.02031092F,-0.04165462F,-0.05353542F,-0.065271154F,-0.100360155F,-0.117233F,-0.06485966F,0.01807445F,0.087694354F,0.07595195F,0.0012445617F,-0.031995527F,-0.006423737F,0.043947805F,0.06873981F,0.038045257F,-0.011480485F,0.0006615822F,0.03804784F,0.026994467F,-0.012714565F,-0.015912049F,0.006200134F,0.01645999F,0.008357543F,-0.020269368F,-0.032873727F,-0.008829556F,0.0155377425F,0.012069063F,0.0029444383F,0.012735432F,0.031228747F,0.03395581F,0.026272345F,0.008449592F,0.0072725257F,0.020340249F,0.010151437F,-0.02133234F,-0.044291757F,-0.05383435F,-0.049624987F,-0.03957869F,-0.043442234F,-0.058206417F,-0.06820663F,-0.011938417F,0.065662935F,0.086030826F,0.04452467F,-0.011961084F,-0.033262245F,-0.00545719F,0.0389434F,0.028903157F,-0.014514869F,-0.03252667F,-0.012273973F,0.016067255F,0.024844615F,0.016103063F,0.02560024F,0.04176124F,0.026665546F,-0.0021254835F,-0.02070143F,-0.011605149F,0.011819306F,0.02056238F,0.004331645F,-0.008008278F,0.014764036F,0.038954798F,0.03548439F,0.014382583F,-0.011951404F,-0.017401345F,-0.022805968F,-0.033740222F,-0.024852516F,-0.0070007965F,0.0043090302F,0.01210228F,-0.007925708F,-0.066335976F,-0.10591732F,-0.10211353F,-0.03897184F,0.046227757F,0.093672246F,0.07858144F,0.03945707F,0.009679945F,-0.0056892717F,0.00064360164F,-0.017204538F,-0.037548512F,-0.022819348F,0.011102217F,0.025390606F,0.012885316F,-0.0007739675F,0.004608395F,0.025602696F,0.036742147F,0.024677472F,0.010568659F,0.017918505F,0.010306166F,-0.013176868F,-0.03275682F,-0.03636773F,-0.0012973249F,0.029658236F,0.034197107F,0.039725613F,0.04311371F,0.029664703F,0.0037792276F,-0.033791147F,-0.05881027F,-0.052775882F,-0.035880268F,-0.029515335F,-0.05120491F,-0.06969044F,-0.076275334F,-0.03968621F,0.043172643F,0.08035559F,0.063667454F,0.027622052F,-0.006153053F,-0.009500679F,0.006967561F,0.0028201807F,-0.013452714F,-0.017471727F,0.0032198215F,0.02626116F,0.01788705F,0.00924202F,0.011752912F,0.02068709F,0.023422942F,-0.0019522156F,-0.021503896F,-0.014599172F,-0.0014927755F,0.01252202F,0.017027123F,0.025494477F,0.04610381F,0.04546234F,0.016064564F,-0.012587648F,-0.0240823F,-0.02219581F,-0.014673574F,-0.0048938664F,-0.006352501F,-0.013329669F,-0.010726615F,-0.02505985F,-0.04299032F,-0.061279304F,-0.078851596F,-0.07168338F,-0.019884149F,0.055974193F,0.084943466F,0.04407687F,-0.002181394F,-0.0064555826F,0.0067514777F,0.033646353F,0.03899164F,0.013024134F,-0.005259293F,-0.008076103F,-0.017118074F,-0.046491098F,-0.048123937F,-0.0005448642F,0.04267949F,0.052021246F,0.031617753F,0.003167729F,-0.00505194F,-0.005323055F,-0.0033058817F,0.006312828F,0.028148608F,0.056772772F,0.058648754F,0.032467652F,0.0070716273F,-0.012429293F,-0.026883427F,-0.036793582F,-0.042819362F,-0.03523901F,-0.017179914F,-0.017805459F,-0.02923446F,-0.05128481F,-0.08635851F,-0.08076234F,-0.03543345F,0.036216043F,0.09291182F,0.09113343F,0.05751056F,0.0155390175F,-0.009128586F,-0.015445397F,-0.02813916F,-0.03901873F,-0.022375667F,0.0011072168F,0.027726587F,0.03485412F,0.013449743F,0.011968017F,0.003658548F,-0.006059667F,-0.0062762313F,-0.0036368573F,0.01632782F,0.040916592F,0.04100696F,0.01752304F,-0.006557001F,-0.02338778F,-0.024980953F,-0.019056613F,0.0023824591F,0.0296985F,0.05035149F,0.05111186F,0.032856125F,-0.012170414F,-0.051164225F,-0.062843725F,-0.06487196F,-0.044556923F,-0.044204243F,-0.05618345F,-0.0578243F,-0.027404409F,0.02358989F,0.042625576F,0.02250815F,0.0028344225F,0.012907891F,0.036982328F,0.053226367F,0.031990726F,0.0021910612F,-0.011471033F,-0.014399544F,-0.010801626F,-0.01065851F,0.0028179798F,0.023148146F,0.032120913F,0.02762829F,0.0046955547F,-0.016260548F,-0.012376944F,-0.002245109F,-0.0039295387F,0.0035974742F,0.019662678F,0.019758234F,0.009025159F,0.009402096F,0.009725851F,0.011586072F,0.022704884F,0.0161855F,0.009488318F,-0.00046682917F,-0.022794671F,-0.04647281F,-0.062181033F,-0.07286469F,-0.08331327F,-0.0795678F,-0.03787801F,0.029411517F,0.06312741F,0.06444931F,0.032360606F,0.0018326566F,0.0036236988F,0.0055100042F,-0.0015359968F,-0.0015422623F,0.008877872F,0.019134296F,0.02760901F,0.013404815F,-0.0021371031F,0.0010079434F,0.0013985503F,0.004132917F,0.0037234628F,0.006488192F,0.017183622F,0.011926972F,-0.0027384786F,-0.010873309F,-0.008376474F,0.0016971622F,0.01633776F,0.019042222F,0.025059372F,0.03455862F,0.025675163F,0.0074570887F,-0.0009871232F,-0.00895592F,-0.02058047F,-0.009154781F,-0.016803505F,-0.037443936F,-0.05337771F,-0.09363791F,-0.1238199F,-0.070046425F,0.013403163F,0.052299004F,0.07323667F,0.06475203F,0.043416653F,0.03378366F,0.01333836F,-0.021136554F,-0.03779265F,-0.020323245F,0.011376106F,0.026177827F,0.028805582F,0.033367895F,0.030158103F,0.016957194F,-0.001725493F,-0.028164012F,-0.035203964F,-0.01651284F,-0.009087437F,-0.00813162F,0.004814541F,0.027753033F,0.035944242F,0.02554468F,0.020126434F,0.019716125F,0.02002866F,0.017520355F,0.011899732F,-0.0054191044F,-0.021987215F,-0.027898228F,-0.049536273F,-0.053712897F,-0.049352504F,-0.06733571F,-0.07581097F,-0.033857927F,0.01123869F,0.03190703F,0.051313456F,0.025198378F,0.0057043647F,0.020755535F,0.025658078F,0.01508096F,0.008405365F,0.0069298353F,0.013392701F,0.015568901F,0.0068107527F,0.018112268F,0.02576536F,0.030114755F,0.02976605F,0.0029023197F,-0.020058922F,-0.025890278F,-0.034043364F,-0.033638306F,-0.017775126F,0.010072759F,0.036594838F,0.04454284F,0.043482125F,0.03259913F,0.015982218F,0.009871483F,0.0016342821F,-0.0070240647F,-0.0075640595F,-0.0137896575F,-0.02820701F,-0.04415147F,-0.06808829F,-0.09145698F,-0.09265283F,-0.054049104F,0.018189136F,0.053693525F,0.055046268F,0.04578829F,0.018677907F,0.006870051F,0.0149998665F,0.0035179497F,-0.014397284F,0.0022650105F,0.01270058F,0.0061501423F,0.0016918737F,0.005149034F,0.016356317F,0.02759084F,0.038386792F,0.027524514F,0.009767663F,0.0019508158F,-0.006936683F,-0.021947576F,-0.03407958F,-0.016241947F,0.0074862307F,0.010224527F,0.014307506F,0.021101054F,0.014612832F,0.014085694F,0.013106301F}};
        int size = audioData.length;
        short audioSample;
        byte[] data = new byte[size*2];
        float[][] testBuf = new float[1][INPUT_TENSOR_SIZE];
        float[][] floatOutputBuffer = new float[1][OUTPUT_DATA_LENGTH];
//        float[][][] floatOutputBuffer = new float[1][129][8];
//        float[][][] floatOutputBuffer = new float[1][129][8];
//        for (int i = 0; i < INPUT_BUFFER_SIZE; i++) {
//            floatInputBuffer[0][i+256] = audioData[i];
//        }

//        Object[] inputArray = {test_input};
        Object[] inputArray = {floatInputBuffer};
        Object[] outputArray = {floatOutputBuffer};
        Map<Integer, Object> outputMap = new HashMap<>();
//        outputMap.put(0, floatOutputBuffer);
        outputMap.put(0, floatOutputBuffer);

        long currentThreadTimeMillisStart = System.currentTimeMillis();

        // Run the model.
        if (isActivated) {
            for (int i = 0; i < OUTPUT_DATA_LENGTH; i++) {
                floatInputBuffer[0][i+256] = audioData[i];
//                testBuf[0][i+256] = audioData[i];
            }
//            System.arraycopy(audioData, 0, floatInputBuffer[0], 256, 896);
            tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
//            for (int i = 0; i < OUTPUT_DATA_LENGTH; i++) {
//                floatOutputBuffer[0][i] = floatInputBuffer[0][i+256];
//            }
//            tfLite.run(test_input, floatOutputBuffer);
        }
        long currentThreadTimeMillisEnd = System.currentTimeMillis();
//        System.arraycopy(audioData, 640, floatInputBuffer[0], 0, 256);
        for (int i = 0; i < 256; i++) {
            floatInputBuffer[0][i] = audioData[640+i];
        }
        double elapsedSeconds = (currentThreadTimeMillisEnd - currentThreadTimeMillisStart);
        if (isActivated) {
            sum += elapsedSeconds;
            count += 1;
        }
        Log.d(TAG, "onAudioDataAvailable: "+ sum / count);
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