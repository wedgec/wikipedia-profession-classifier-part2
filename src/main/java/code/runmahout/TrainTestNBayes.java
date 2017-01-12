/* TrainTestNBayes - This class trains and tests naive bayes 
 * model -- uses Mahout software.
 */


package code.runmahout;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.TreeSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import code.vectorcreate.CreateVectorMapred;

/* custom class--double prediction represents probability according to model that
 * vector corresponds to the profession represented by int index. index is the location
 * of the coupled profession within in a (separate) sorted list. Implements 
 * comparable so we can sort by prediction, thus allowing us to identify the most
 * probable professions for a given test vector*/
class PredictionIndexPair implements Comparable <PredictionIndexPair>{
	private double prediction;
	private int index;
	
	PredictionIndexPair(double prediction, int index){
		this.prediction = prediction;
		this.index = index;
	}
	public double getPrediction(){
		return prediction;
	}
	public int getIndex(){
		return index;
	}
	// allows sorting in descending order of predictions
	public int compareTo(PredictionIndexPair p) {
		if (this.getPrediction() < p.getPrediction())
			return 1;
		else if(this.getPrediction() > p.getPrediction())
			return -1;
		else
			return 0;
	}
	public String toString(){
		return "[ "+this.prediction + ", "+this.index+ "]";
	}
}

//class for training and testing model
public class TrainTestNBayes {	
	//controls printing of debug statements
	private static final boolean DEBUG = false;

	public static void main(String[] args) throws Exception {
		boolean train = false;
		if(args.length == 1){
			if (args[0].equals("1")){
				System.out.println("Will train too.");
				train = true;
			}
		}
		File predictionsFile = new File("prediction-results.txt");
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		
		TrainNaiveBayesJob trainNBayes = new TrainNaiveBayesJob();
		trainNBayes.setConf(conf);
		
		//path to local disk folders create by CreateVectorMapred class
		String inputFile = CreateVectorMapred.TRAIN_MAHOUT_SEQFILE_PATH;  
		String testVectorsFile = CreateVectorMapred.TEST_MAHOUT_SEQFILE_PATH;
		//Our model will be stored here
		String outputDirectory = "pa3TainingOutput";
		String tempDirectory = "pa3TrainingTmp";
		
		if(train) {
			fs.delete(new Path(outputDirectory), true);
			fs.delete(new Path(tempDirectory), true);		
			
			System.out.println("Training with NBayes...");
			trainNBayes.run(new String[] { "--input", inputFile, "--output", outputDirectory, "-el", "--overwrite", "--tempDir", tempDirectory, "-li", "labelIndexes"});
			System.out.println("Done...");
		}
		
		// Train the classifier
		NaiveBayesModel naiveBayesModel = NaiveBayesModel.materialize(new Path(outputDirectory), conf);
		System.out.println("Features: " + naiveBayesModel.numFeatures());
		System.out.println("Labels: " + naiveBayesModel.numLabels());
		
		//Create the classifier
		AbstractVectorClassifier classifier = new ComplementaryNaiveBayesClassifier(naiveBayesModel);
		int numClasses = classifier.numCategories();
		
		//list of professions sorted
		List<String> sortedProfessions;
		TreeSet<String> professionsSet = new TreeSet<String>();
		
		//read from inputFile
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(inputFile), conf);
		Text keyTxt = new Text();
		System.out.println("Creating professions list...");
		//add professions to list
		while (reader.next(keyTxt)) {
			String[] prof = keyTxt.toString().split("\\/");
			professionsSet.add(prof[1]);
		}
		reader.close();
		sortedProfessions = new ArrayList<String>(professionsSet);
		System.out.println("Done.");
		
		if(DEBUG){
			int i=0;
			for(String profession: sortedProfessions) {
				System.out.println(i + ": " + profession);
				i++;
			}
		}
		
		//test prints
		System.out.println("Model Labels: " + naiveBayesModel.numLabels());
		System.out.println("Professions Hash Set size: "+ professionsSet.size() + "\nProfessions List Size: "+ sortedProfessions.size());
		
		//run classifier
		SequenceFile.Reader sfReader = new SequenceFile.Reader(fs, new Path(testVectorsFile), conf);
		ArrayList<PredictionIndexPair> probabilitiesList;
		
		Text keyText = new Text();
		VectorWritable valueVecWritable = new VectorWritable();
		
		int correctPredictions = 0;
		int totalPredictions = 0;
		// that's the number of test vectors!
		int expectedVectors = 133417;
		
		PrintWriter writer = new PrintWriter(predictionsFile);
		
		while (sfReader.next(keyText, valueVecWritable)) {
			probabilitiesList = new ArrayList<PredictionIndexPair>();
			String[] articleAndProffInfo = keyText.toString().split(":::");
			String articleTitle = articleAndProffInfo[0];
			totalPredictions++;
			if (totalPredictions % 2000 == 0){
				System.out.println("Predicted " + (int) (((double)totalPredictions / (double)expectedVectors ) * 100.0) + "%");
				System.out.print(correctPredictions +" correct predictions so far: ");
				System.out.printf("%.2f%% accuracy\n", ((double)correctPredictions/totalPredictions) * 100.00);
			}
			String[] professions = articleAndProffInfo[1].split(",");
			List<String> actualProfessions = new ArrayList<String>(Arrays.asList(professions));
			Vector vec = valueVecWritable.get();
			Vector prediction = classifier.classifyFull(vec);
			if(DEBUG)
				System.out.println("Number of classes (classifier.numCategories() ): "+numClasses);
			for (int i = 0; i < numClasses; i++) {
				probabilitiesList.add(new PredictionIndexPair(prediction.get(i), i));
				if(DEBUG)
					System.out.println("Adding key: "+prediction.get(i) +", val: "+i+ ", Profession in that index from sorted professions: "+sortedProfessions.get(i));
			}
			ArrayList<String> bestThreeProfs = new ArrayList<String>();
			Collections.sort(probabilitiesList);
			for (int i = 0; i < 3; i++) {
				bestThreeProfs.add(sortedProfessions.get(probabilitiesList.get(i).getIndex()));
			}
			
			String entry = articleTitle + " : ";
			int i =0;
			for (String profession: bestThreeProfs) {
				entry += profession + (i==2? "" : ", " );
				i++;
			}
			writer.println(entry);
			boolean correctPrediction = false;
			if(DEBUG)
				System.out.print("Actual Professions : ");
			
			for (String profession: actualProfessions) {
				if (DEBUG)
					System.out.println(profession + " " );
				
				if (!correctPrediction && bestThreeProfs.contains(profession)) {
					correctPrediction = true;
					correctPredictions++;
				}
			}
		}
		sfReader.close();
		writer.close();
		System.out.print("Percent correct predictions : ");
		System.out.printf("%.2f%%\n", ((double)correctPredictions/totalPredictions) * 100.00);
	    
	}
}
