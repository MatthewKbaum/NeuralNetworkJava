import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Source {
	public static int minWordLength = 5;
	
	public static void main(String[] args) {
		System.out.println("Retrieving MNIST data...");
		//Load training data
		MNISTReader reader = new MNISTReader();
		List<float[]> training_images = reader.getImages("C:\\Users\\mwkba\\Desktop\\Programs\\Java\\MNISTClassifier\\src\\train-images.idx3-ubyte");
		int[] training_labels = reader.getLabels("C:\\Users\\mwkba\\Desktop\\Programs\\Java\\MNISTClassifier\\src\\train-labels.idx1-ubyte");
		
		//Load test data
		List<float[]> test_images = reader.getImages("C:\\Users\\mwkba\\Desktop\\Programs\\Java\\MNISTClassifier\\src\\t10k-images.idx3-ubyte");
		int[] test_labels = reader.getLabels("C:\\Users\\mwkba\\Desktop\\Programs\\Java\\MNISTClassifier\\src\\t10k-labels.idx1-ubyte");
		System.out.println("Formatting MNIST data...");
		
		//Now combine the data into one array, so it can formatted easily
		List<float[]> images = new ArrayList<float[]>();
		images.addAll(training_images);
		images.addAll(test_images);
		int[] labels = new int[training_labels.length + test_labels.length];
		for(int i = 0; i < training_labels.length; i ++){
			labels[i] = training_labels[i];
		}
		for(int i = 0; i < test_labels.length; i ++){
			labels[training_labels.length + i] = test_labels[i];
		}

		//Format all of the data
		TrainingExample[] trainingSet = new TrainingExample[training_images.size()];
		TrainingExample[] testSet = new TrainingExample[test_images.size()];
		for(int i = 0; i < images.size(); i ++){
			//Convert labels from integer to array ( 0-9 to [1,0,0,0,0,0,0,0,0,0] )
			float[] outputArr = new float[10];
			for(int j = 0; j < outputArr.length; j ++){
				outputArr[j] = 0;
			}
			outputArr[labels[i]] = 1;
			
			//Normalize float[] to between 0 and 1
			for(int j = 0; j < images.get(i).length; j ++){
				images.get(i)[j] /= 255;
			}
			
			//Seperate the data between training data and test data'
			if(i < trainingSet.length){
				trainingSet[i] = new TrainingExample(images.get(i), outputArr);
			}
			else{
				testSet[i - trainingSet.length] = new TrainingExample(images.get(i), outputArr);
			}
		}
		

		int[] size = new int[]{784,30,30,10};
		Network net = new Network(size, 1, 0.2f, 1f);
		
		int totalEpochs = 100;
		int epochRate = 1;
		
		System.out.println("Starting analysis of data...");
		int epochOn = 0;
		while(epochOn <= totalEpochs){
			System.out.println("----");
			System.out.println("Epoch: "+epochOn);
			System.out.println("Training: " + net.testData(trainingSet)/training_images.size() * 100 + "%");
			System.out.println("Test: "+net.testData(testSet)/test_images.size() * 100 + "%");
			net.saveWeights("weights.txt");
			net.saveBiases("biases.txt");
			net.stochasticGradientDescent(trainingSet, epochRate);
			epochOn += epochRate;
		}
		System.out.println("Analysis complete. Final results:");
		System.out.println(net.testData(testSet)/test_images.size() * 100 + "%");

		net.saveWeights("weights.txt");
		net.saveBiases("biases.txt");
		System.out.println("Results have been saved to weights.txt and bias.txt");
	}
}