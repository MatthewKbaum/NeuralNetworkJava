import java.util.Random;

public class Neuron {
	float[] inputs;
	float[] weights;
	float bias;
	float output;
	float tempGradient;
	float totalGradient;
	public Neuron(float[] inputs){
		this.inputs = inputs;
		bias = (float) Math.random();
		generateWeights();
	}
	
	public void generateWeights(){
		weights = new float[inputs.length];
		//Initialize the weights with random values
		Random r = new Random();
		for(int i = 0; i < weights.length; i ++){
			weights[i] = (float) r.nextGaussian();
		}
	}
	
	//Update the output to match the inputs/weights
	public void update(){
		output = 0;
		for(int i = 0; i < weights.length; i ++){
			output += (weights[i] * inputs[i]);
		}
		output = sigmoid(output + bias);
	}
	
	private float sigmoid(float z){
		return (float) (1.0 / (1.0 + Math.exp(-z)));
	}
}
