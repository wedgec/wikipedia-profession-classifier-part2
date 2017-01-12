/* Group:      Hadoop05 
 * Email:      wedgeco@brandeis.edu, tsikudis@brandeis.edu, dimos@brandeis.edu, 
 * Course:     COSI 129a
 * Assignment: PA3  
 * 
 * Program:    VectorCreateMapred - Runs a mapreduce (no reducer needed) job to
 * 			   create Sequence File  of feature vectors associated
 *             with various professions. Vectors will be used to train
 *             and test a Naive Bayes model. This class is used to 
 *             build the vectors of both the training set and test set
 *             (training and test set vectors are built one run at a time,
 *             they are formatted differently, and they require different 
 *             command parameters). Original data comes from Wikipedia articles. 
 */


package code.vectorcreate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;


public class CreateVectorMapred{
	//Paths to store vector in local disk. TrainTestNBayes class that run mahout reads from local disk
	public static final String TRAIN_MAHOUT_SEQFILE_PATH = "pa3TrainVectors/trainvectors";
	public static final String TEST_MAHOUT_SEQFILE_PATH = "pa3TestVectors/testvectors";
	
	public static class CreateVectorMapper extends Mapper<LongWritable, Text, Text, VectorWritable> {
		private static final String PROFESSIONS_FILE = "professions.txt";
		private Map<String, List<String>> professionsMap;
		//these maps map lemmas to length 2 information arrays
		private Map<String, int[]> trainingLemmaInfoMap;
		//length of all feature vectors
		private int numTotalFeatures;
		//these fields set by extra arguments in user's command
		boolean isTestSet;
		private String trainingLemmasFile;
		private int numDocuments;
		
		//parses files and builds relevant data structures prior to map function
		@Override
		protected void setup(Mapper<LongWritable, Text, Text, VectorWritable>.Context context)
				throws IOException, InterruptedException {
			super.setup(context);
			
			//sets fields based on extra arguments entered in command--returns false upon error
			 if (!setFieldsBasedOnParameters(context)) {
				 System.err.println("Something went wrong with extra arguments.");
				 return;
			 }
			
			//parses professions file and builds map from names to lists of professions--returns false upon error
			if (!parseProfessionsFile()) {
				System.err.println("Something went wrong with building the profession list.");
				return;
			}
			
			/* builds map of training set lemmas--each lemma is mapped to array containing vector index and 
			 * document frequency. Returns null upon error*/
			if ( (trainingLemmaInfoMap = buildLemmaInfoMap()) == null ) {
				System.err.println("Something went wrong with building the lemma info map");
				return;
			}
		}
		
		/* builds map of lemmas to length 2 info arrays--cell 0 stores lemma's vector index and cell 1 stores lemma's 
		 * document frequency*/
		private Map<String, int[]> buildLemmaInfoMap() {
			
			Map<String, int[]> lemmaInfoMap = new HashMap<String, int[]>();
			//HDFS path--this file stores lemma, document frequency pairs
			Path filePath=new Path("hdfs:" + trainingLemmasFile); //should look like "hdfs:/path/to/file"
			//Hadoop FileSystem class
            FileSystem fs;
			
            try {
				fs = FileSystem.get(new Configuration());
				BufferedReader input = new BufferedReader(new InputStreamReader(fs.open(filePath), StandardCharsets.UTF_8));
				
				//index that represents where lemma will be stored in feature vector--each lemma gets unique integer
				int featureIndex = -1;
				
				while(input.ready()){
					//split key and value by tab
					String[] lineContents = input.readLine().split("\\t");
					String lemma = lineContents[0];
										
					//if somehow identical lemmas have been read in from file, skips all entries after the first
					if (lemmaInfoMap.containsKey(lemma)) {
						System.err.println("Vocab has non-unique lemmas: " + lemma);
						continue;
					}
					
					int docFreq = Integer.valueOf(lineContents[1]);
					featureIndex++;
					
					lemmaInfoMap.put(lemma, new int[] {featureIndex, docFreq});
					
				}
				input.close();
				
				//sets total size for all feature vectors
				numTotalFeatures = lemmaInfoMap.size();
				
				System.out.println("Okay, parsed "+ trainingLemmasFile + " with " 
						+ lemmaInfoMap.size() + " number of lemmas.");
								
				return lemmaInfoMap;
				
			} catch(IOException ioe){
				System.err.println("Error while reading from file " + trainingLemmasFile);
				ioe.printStackTrace();
				return null;
			}
		}

		//parses professions file and builds map from names to lists of professions--returns false upon error
		private boolean parseProfessionsFile() {
			InputStreamReader isr;
			// this HashMap will hold the peoples' names
			professionsMap = new HashMap<String, List<String>>();
			InputStream is = this.getClass().getClassLoader().getResourceAsStream(PROFESSIONS_FILE);
			
			if (is == null){
				System.err.println("Error while getting resource from "+ PROFESSIONS_FILE);
				return false;
			}
			// read from within the JAR
			isr = new InputStreamReader(is, StandardCharsets.UTF_8);
			BufferedReader input = new BufferedReader(isr);
			String line, name, professions;
			List<String> professionsList;
			//for figuring out average number of lemmas per article
			try {
				while(input.ready()){
					line = input.readLine();
					int indexToSplit = line.lastIndexOf(':');
					name = line.substring(0, indexToSplit).trim();
					professions = line.substring(indexToSplit+1).trim();
					professionsList = new ArrayList<String>();
					for(String prof : professions.split(",")){
						professionsList.add(prof.trim());
					}
					professionsMap.put(name, professionsList);
				}
				input.close();
			} catch(IOException ioe){
				System.err.println("Error while reading from file " + PROFESSIONS_FILE);
				ioe.printStackTrace();
				return false;
			}
			System.out.println("Okay, parsed "+ PROFESSIONS_FILE + " with "
					+ professionsMap.size() + " number of professions.");
			return true;
		}

		//sets fields based on extra arguments entered in command--returns false upon error
		private boolean setFieldsBasedOnParameters(Context context) {
			//job runs as training set or test set job
			if (context.getConfiguration().get("type").equals("train")) {
				isTestSet = false;
			} else if (context.getConfiguration().get("type").equals("test")) {
				isTestSet = true;
			} else {
				System.err.println("\"type\" parameter must be 'train' or 'test'");
				return false;
			}
			//training set lemmas
			trainingLemmasFile = context.getConfiguration().get("trainingLemmasPath");
			try {
				//number of documents in set (training or test) for which vectors are being built
				numDocuments = Integer.parseInt(context.getConfiguration().get("numTrainingDocs"));
			}
			catch (NumberFormatException nfe){
				System.err.println("Wrong number!");
				nfe.printStackTrace();
				return false;
			}
			
			System.out.println("==========Configuration==========");
			if(isTestSet)	System.out.println("Type: Test");
			else			System.out.println("Type: Train");
			System.out.println("Training lemmas path: " + trainingLemmasFile);
			System.out.println("Number of documents: " + numDocuments);
			System.out.println("=================================");
			
			return true;
		}
		
		/* map function--takes article name, lemma index pair and outputs feature vectors as value. 
		 * For training set run, key outputted is profession embedded in forward slashes. For test set 
		 * run key includes article name and all associated professions. TF-IDF is used for feature values*/
		@Override
		public void map(LongWritable lineNum, Text lemmaFreqs, Context context)
				throws IOException, InterruptedException {
			
			//estimate 100 features with non-zero values per article, for purposes of optimization
			Vector vector = new RandomAccessSparseVector(numTotalFeatures, 100);
			
			//separate article title from lemma index by tab--array should be size 2
			String[] allContent = lemmaFreqs.toString().split("\\t");
			if(allContent.length < 2){
				System.err.println("Something went wrong with splitting lemmaFreqs in map. "  + lemmaFreqs.toString());
			}
						
			//just in case article name contains a tab, reconstruct it:
			String articleName = "";
			for (int i = 0; i < allContent.length - 1; i++) {
				articleName += allContent[i] + "\t";
			}
			articleName = articleName.trim();
			
			String entireLemmaIndex = allContent[allContent.length - 1];
			List<String> professions;
			
			if(entireLemmaIndex.indexOf('<') == -1){
				System.err.println("Document has no words!");
			} else if ((professions = professionsMap.get(articleName)) == null) {
				/* do nothing--article has no associated professions. We exclude articles with no professions
				 * not only for training vectors, but also for test vectors, since without associated professions
				 * we have no way to test predictions for that vector*/
				System.out.println("Article: " + articleName + " does not have any associated professions");
			} else {								
				//output objects
				VectorWritable vectorWritable = new VectorWritable();
				Text professionText = new Text();
				
				//adds one entry to vector per iteration
				for (String lemmaFreq: entireLemmaIndex.split(">,")){
					//string now lemma and count separated by comma
					String lemmaFreqCleaned = lemmaFreq.replaceAll(">|<", "");
					
					//use last comma as delimiter, since lemma could contain comma
					int lastCommaIndex = lemmaFreqCleaned.lastIndexOf(',');
					String lemma = lemmaFreqCleaned.substring(0, lastCommaIndex);
					int termFreq = Integer.valueOf(lemmaFreqCleaned.substring(lastCommaIndex + 1));
					
					//gets array that stores feature index and doc frequency
					int[] lemmaInfo = trainingLemmaInfoMap.get(lemma);
					
					if (lemmaInfo == null) {
						if (!isTestSet) {
							System.err.println("Lemma " + lemma + " did not map to an info array!");
							return;
						} else {
							//for test set, we simply exclude all out of vocabulary lemmas
							System.out.println("(MAP) Lemma " + lemma + " not in vocabulary, we can skip");
							continue;
						}	
					}
					int featureIndex = lemmaInfo[0];
					int docFreq = lemmaInfo[1];
					//calculates TF-IDF and sets result to vector
					double idf = Math.log10((double)numDocuments / docFreq);
					double tfIDF = termFreq * idf;
					vector.set(featureIndex, tfIDF);
						
				}
				//sets vector to vectorWritable
				vectorWritable.set(vector);
				
				/* if test set run, key is of format 'articleName:::profession1,profession2...professionN'--The
				 * idea is to couple the article name with its professions to make evaluation convenient*/
				if(isTestSet){
					String professionsString = "";
					int i=0;
					for (String profession : professions) {
						if (i == 0) {
							professionsString += ":::";
						}
						professionsString += profession + (i== professions.size()-1? "" : ",");
						i++;
					}
					//test set run output
					context.write(new Text(articleName + professionsString), vectorWritable);
				}else{
					//if training, for each document we output one vector per profession
					for (String profession : professions) {
						professionText.set("/" + profession + "/");
						//training set run output
						context.write(professionText, vectorWritable);
					}
				}
			}
		}
	}
	
	//controls mapreduce job
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		/* command requires additional parameters as follows:
		 * type = 'train' or 'test'
		 * trainingLemmasPath = PATH TO TRAINING SET LEMMAS/IDFs (hdfs:/ + PATH = complete path)
		 * numTrainingDocs = NUMBER OF DOCUMENTS IN THE TRAINING SET  
		 */
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 2){
			System.err.println("Usage: create-vector-mapred <in> <out>");
			System.exit(2);
		}
		Job job = Job.getInstance(conf, "Create Mahout vector");
		job.setJarByClass(CreateVectorMapred.class);
		job.setMapperClass(CreateVectorMapper.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		boolean finishedOK = job.waitForCompletion(true);
		// copy the HDFS file containing vectors to a local file, since Mahout reads from local!
		FileSystem fs = FileSystem.newInstance(conf);
		if (conf.get("type").equals("train"))
			fs.copyToLocalFile(new Path(otherArgs[1], "part-r-00000"), new Path(TRAIN_MAHOUT_SEQFILE_PATH));
		else if (conf.get("type").equals("test"))
			fs.copyToLocalFile(new Path(otherArgs[1], "part-r-00000"), new Path(TEST_MAHOUT_SEQFILE_PATH));
		fs.close();
		System.exit( finishedOK? 0 : 1); 
	}
}
