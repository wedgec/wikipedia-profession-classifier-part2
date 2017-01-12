/* ExportLemmasMapred - This class runs a mapreduce 
 * job. It takes as input a file of articleName, lemmaIndex<lemma, count>
 * pairs, and outputs a file of lemma, documentFrequency pairs. The output not
 * only gives us document frequencies for every lemma, but also defines
 * our vocabulary. Note that we exclude from output all lemmas that have
 * a document frequency of 1, on the assumption that these lemmas are
 * mostly erroneous/junk lemmas.
 */

package code.vectorcreate;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class ExportLemmasMapred {
	
	public static class ExportLemmasMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
		Text outputKey;
		IntWritable outputVal;
		
		//outputs lemma, count=1 pairs
		@Override
		public void map(LongWritable lineNum, Text docLine, Context context)
				throws IOException, InterruptedException {
			//get the string with lemmas and their frequencies discarding the title
			String docLineStr = docLine.toString();
			if( docLineStr.indexOf('<') == -1){
				System.err.println("Document has no words!");
			}else{
				outputKey = new Text();
				outputVal = new IntWritable();
				String lemmaFreqsStr = docLineStr.substring(docLineStr.indexOf('<')+1);
				// single pass in the lemma-frequencies line by splitting with ">,".
				for (String lemmaFreq: lemmaFreqsStr.split(">,")){
					String lemmaFreqCleaned = lemmaFreq.replaceAll(">|<", "");
					String lemma = lemmaFreqCleaned.substring(0, lemmaFreqCleaned.lastIndexOf(','));
					outputKey.set(lemma);
					outputVal.set(1);
					context.write(outputKey, outputVal);
				}
			}
		}
	}
	
	//outputs lemma, documentFrequency pairs
	public static class ExportLemmasReducer extends Reducer <Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(Text lemma, Iterable<IntWritable> DFs, Context context)
				throws IOException, InterruptedException {
			Text outKey;
			IntWritable outVal;
			
			outKey = new Text();
			outVal = new IntWritable();
			outKey.set(lemma);
			int df=0;
			for(IntWritable dfWritable : DFs){
				df+=dfWritable.get();
			}
			outVal.set(df);
			
			if(df != 1) {context.write(outKey, outVal);}
			else {} //exclude from vocab words that appear in only one document
			
		}
	}
	
	//runs mapreduce job
	public static void main(String[] args) throws Exception{

		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		
		if (otherArgs.length != 2){
			System.err.println("Usage: export-lemmas-df-mapred <in> <out>");
			System.exit(2);
		}
		
		Job job = Job.getInstance(conf, "Export lemmas and DFs");
		job.setJarByClass(ExportLemmasMapred.class);
		job.setMapperClass(ExportLemmasMapper.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setCombinerClass(ExportLemmasReducer.class);
		job.setReducerClass(ExportLemmasReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);		
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
