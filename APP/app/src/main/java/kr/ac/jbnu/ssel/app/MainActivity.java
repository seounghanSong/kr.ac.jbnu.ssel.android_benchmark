package kr.ac.jbnu.ssel.app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize the python interpreter
        initPython();

        ArrayList<InputStream> imagesFromAsset = new ArrayList<>();
        ArrayList<ArrayList<Rect>> detectedFacesAll = new ArrayList<>();

        try {
            // Retrieving all the bitmap under assets/images folder
            String[] images = getAssets().list("test_images");
            assert images != null;

            for (String image_name : images) {
                imagesFromAsset.add(getAssets().open("test_images/" + image_name));
            }

            for (InputStream inputStream : imagesFromAsset) {
                // ============================== System Operation Time Check
                long startTime = System.nanoTime();

                Log.d("inputStream", inputStream.toString());
                detectFaces(inputStream);

                long endTime = System.nanoTime();
                // ============================== System Operation Time Check

                long lTime = endTime - startTime;
                Log.d("Operation Time", (lTime/1000000.0) + "(ms)");
            }
        } catch (IOException e) {
            Log.e("Pytorch Mobile", "Error while reading assets", e);
            finish();
        }
    }

    private void initPython() {
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
    }

    private void detectFaces(InputStream inputStream) {
        ArrayList<Tensor> tensorList = new ArrayList<>();
        int height;
        int width;

        // Creating bitmap from the inputStream
        Bitmap bitmap = BitmapFactory.decodeStream(inputStream);

        // Showing image on screen
        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(bitmap);

        height = bitmap.getHeight();
        width = bitmap.getWidth();

        try {
            IValue[] detectResult = detect(bitmap, "Retinaface_mobilenetV1_mobile.pt");

            for (IValue iValue : detectResult) {
                Tensor tensor = iValue.toTensor();
                tensorList.add(tensor);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (tensorList.size() == 3) {
            Python py = Python.getInstance();
            final PyObject pyObject = py.getModule("script");

            // Check the elements within the tensor
            float[] locFloatArray = tensorList.get(0).getDataAsFloatArray();
            float[] confFloatArray = tensorList.get(1).getDataAsFloatArray();
            // ====================================

            pyObject.callAttr("main", locFloatArray, confFloatArray, height, width);
        }

        // Pause
//        if (tensorList.size() == 3) {
//            // TODO: Running the Python script
//            Python py = Python.getInstance();
//            final PyObject pyObject = py.getModule("script");
//
//            PyObject pyobj = pyObject.callAttr("main", tensorList.get(0), tensorList.get(1), height, width);
//            Log.d("detection_result", pyobj.toString());
//        }
    }

    private IValue[] detect(Bitmap bitmap, String model_name) throws IOException {
        // Loading serialized torchscript module (Loading the model)
        // Ex) app/src/main/assets/{model_name}.pt
        Module module = Module.load(assetFilePath(this, model_name));

        // Preparing input tensor
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        // Do inference by running the model.
        // final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        final IValue[] outputIValue = module.forward(IValue.from(inputTensor)).toTuple();

        return outputIValue;
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;

                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}