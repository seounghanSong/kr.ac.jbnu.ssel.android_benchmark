package kr.ac.jbnu.ssel.app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ArrayList<InputStream> imagesFromAsset = new ArrayList<>();
        Integer counter = 0;

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

                IValue[] result = detect(inputStream, "Retinaface_mobilenetV1_mobile.pt");
                counter += 1;

                long endTime = System.nanoTime();

                // Total time (end - start)
                long lTime = endTime - startTime;
                Log.d("Operation Time", (lTime/1000000.0) + "(ms)");
                Log.d("Operation Order", counter + "(unit)");

                // ============================== Model Inference Result Check

                Log.d("Inference Length", String.valueOf(result.length));
                for (IValue iValue : result) {
                    if (iValue.isTensor()) {
                        Tensor valueTensor = iValue.toTensor();
                        float[] valueFloat = valueTensor.getDataAsFloatArray();

                        Log.d("Inference Value", String.valueOf(valueFloat));
                    }
                }
            }

        } catch (IOException e) {
            Log.e("Pytorch Mobile", "Error while reading assets", e);
            finish();
        }
    }

    private IValue[] detect(InputStream inputStream, String model_name) throws IOException {
        Bitmap bitmap = null;
        Module module = null;

        // Creating bitmap from the inputStream
        bitmap = BitmapFactory.decodeStream(inputStream);

        // Showing image on screen
        ImageView imageView = findViewById(R.id.image);
        imageView.setImageBitmap(bitmap);

        // Loading serialized torchscript module (Loading the model)
        // Ex) app/src/main/assets/{model_name}.pt
        module = Module.load(assetFilePath(this, model_name));

        // Preparing input tensor
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        // Do inference bty running the model.
        // final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        final IValue[] outputIValue = module.forward(IValue.from(inputTensor)).toTuple();

        return outputIValue;
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
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