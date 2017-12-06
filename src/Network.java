import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collections;

public class Network {
	int[] size;
	int miniBatchSize;
	float learningRate;
	float lambda;
	Neuron[][] layers;
	public Network(int[] size, int miniBatchSize, float learningRate, float lambda){
		this.miniBatchSize = miniBatchSize;
		this.learningRate = learningRate;
		this.lambda = lambda;
		this.size = size;
		layers = new Neuron[size.length][1];
		//Fit the layer to the appropriate size
		for(int i = 0; i < size.length; i ++){
			layers[i] = new Neuron[size[i]];
		}
		
		//Generate weights for the neurons
		for(int i = 0; i < layers.length; i ++){
			for(int j = 0; j < layers[i].length; j ++){
				if(i > 0){
					layers[i][j] = new Neuron(new float[layers[i-1].length]);
				}
				else{
					layers[i][j] = new Neuron(new float[0]);
				}
			}
		}
	}
	
	//Feed forward the network given a set of inputs (run through the network, generate outputs)
	public void feedForward(float[] inputs){
		//Put the inputs into the system
		for(int i = 0; i < layers[0].length; i ++){
			layers[0][i].output = inputs[i];
		}
		//Loop through the system, setting the inputs/outputs correctly
		for(int i = 1; i < layers.length; i ++){
			//Create a list of outputs from the previous layer in order to use them as the inputs to the current layer
			float[] prev_outputs = new float[layers[i-1].length];
			for(int j  = 0; j < layers[i-1].length; j ++){
				prev_outputs[j] = layers[i-1][j].output;
			}
			//Now set the outputs of the current layer based on those inputs
			for(int j = 0; j < layers[i].length; j ++){
				layers[i][j].inputs = prev_outputs;
				layers[i][j].update();
			}
		}
	}
	
	public void stochasticGradientDescent(TrainingExample[] trainingSet, int epochs){
		for(int i = 0; i < epochs; i ++){
			Collections.shuffle(Arrays.asList(trainingSet));
			//Create mini batches
			TrainingExample[] miniBatch = new TrainingExample[miniBatchSize];
			for(int j = 0; j < trainingSet.length; j ++){
				miniBatch[j % miniBatchSize] = trainingSet[j]; 
				if(j % miniBatchSize == 0 && j != 0){
					//Handle the mini batches
					handleBatch(miniBatch, trainingSet.length);
				}
			}
		}
	}
	
	public void handleBatch(TrainingExample[] miniBatch, int n){
		//Backpropigate everything!
		for(int i = 0; i < miniBatch.length; i ++){
			backpropigate(miniBatch[i]);
		}
		
		//Update weights based on totalGradients from backpropigation
		for(int i = 1; i < layers.length; i ++){
			for(int j = 0; j < layers[i].length; j ++){
				for(int m = 0; m < layers[i][j].weights.length; m ++){
					//This is where the magic happens (aka math)
					layers[i][j].weights[m] = layers[i][j].weights[m]*(1-learningRate*(lambda/n)) - learningRate * (layers[i-1][m].output) * (layers[i][j].totalGradient / miniBatchSize);
				}
				layers[i][j].bias -= learningRate * (layers[i][j].totalGradient / miniBatchSize);
				layers[i][j].totalGradient = 0;
			}
		}
	}
	
	public void backpropigate(TrainingExample example){
		//First, feed forward
		feedForward(example.inputs);
		
		//Then, update the gradients going backwards
		for(int i = layers.length-1; i >= 1; i --){ //Start at end
			for(int j = 0; j < layers[i].length; j ++){
				//Calculate error differently, depending if output or hidden
				float error = 0;
				if(i == layers.length-1){ //Output
					error = (layers[i][j].output - example.outputs[j]);
				}
				else{ //Hidden
					for(int m = 0; m < layers[i+1].length; m ++){
						error += layers[i+1][m].tempGradient * layers[i+1][m].weights[j];
					}
				}
				//Now update the gradients based on the errors
				float gradient = (error) * layers[i][j].output * (1 - layers[i][j].output);
				layers[i][j].tempGradient = gradient;
				layers[i][j].totalGradient += gradient;
			}
		}
	}
	
	//TESTING FUNCTIONS, used to test system
	public float testAnecdote(TrainingExample example){
		feedForward(example.inputs);
		return layers[layers.length-1][0].output;
	}
	
	public float testData(TrainingExample[] testingData){
		int correct = 0;
		for(int i = 0; i < testingData.length; i ++){
			feedForward(testingData[i].inputs);
			//'Correctness' function: Finds if the index of the largest # in the output is the same as the index of the correct largest #
			float[] outputs = new float[layers[layers.length-1].length];
			for(int j = 0; j < layers[layers.length-1].length; j ++){
				outputs[j] = layers[layers.length-1][j].output;
			}
			int outputIndex = largestIndex(outputs);
			int correctIndex = largestIndex(testingData[i].outputs);
			if(outputIndex == correctIndex){
				correct ++;
			}
		}
		return correct;
	}
	
	//SAVE/GET FUNCTIONS
	public void saveWeights(String file){ //This will put the current weights into the inputted file, in the proper format to be retrieved later on
	    PrintWriter writer = null;
		try {
			writer = new PrintWriter(file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		for(int i = 1; i < layers.length; i ++){
			for(int j = 0; j < layers[i].length; j ++){
				//Store in a line as: i,j,weights
				String line = i+","+j;
				for(int m = 0; m < layers[i][j].weights.length; m ++){
					line += "," + layers[i][j].weights[m];
				}
				writer.println(line);
			}
		}
	    writer.close();
	}
	
	public void getWeights(String file) throws IOException { //This will set the weights based off of the inputted file
		BufferedReader reader = new BufferedReader(new FileReader(file));
		String line = reader.readLine();
		while(line != null){
			//Set weights based on the line
			String[] lineSplit = line.split(",");
			for(int i = 2; i < lineSplit.length; i ++){
				layers[Integer.parseInt(lineSplit[0])][Integer.parseInt(lineSplit[1])].weights[i-2] = Float.parseFloat(lineSplit[i]);
			}
			line = reader.readLine();
		}
	}
	
	public void saveBiases(String file){
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		for(int i = 1; i < layers.length; i ++){
			for(int j = 0; j < layers[i].length; j ++){
				//Store in a line as: i,j,bias
				String line = i+","+j+","+layers[i][j].bias;
				writer.println(line);
			}
		}
	    writer.close();
	}
	
	public void getBiases(String file) throws IOException{
		BufferedReader reader = new BufferedReader(new FileReader(file));
		String line = reader.readLine();
		while(line != null){
			//Set bias based on the line
			String[] lineSplit = line.split(",");
			layers[Integer.parseInt(lineSplit[0])][Integer.parseInt(lineSplit[1])].bias = Float.parseFloat(lineSplit[2]);
			line = reader.readLine();
		}
	}
	
	//Function to find largest # in array and return index
	private int largestIndex(float[] arr){
		float largest = 0;
		int largestIndex = 0;
		for(int i = 0; i < arr.length; i ++){
			if(arr[i] >= largest){
				largestIndex = i;
				largest = arr[i];
			}
		}
		return largestIndex;
		
	}
	
}