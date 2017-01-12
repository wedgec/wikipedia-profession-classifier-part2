package code.runmahout;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

public class ReadPredictionResult {
	public static void main(String[] args) {
		if (args.length != 1){
			System.err.println("Usage: ReadPredictionResult <predictionFilePath>");
			System.exit(1);
		}
		String filePath = args[0];
		Configuration conf = new Configuration();
		try {
			FileSystem fs = FileSystem.get(conf);
			Text keyTxt = new Text();
			SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(filePath), conf);
			while (reader.next(keyTxt)) {
				System.out.println(keyTxt.toString());
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
