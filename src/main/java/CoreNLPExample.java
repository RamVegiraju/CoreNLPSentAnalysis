import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations.SentimentAnnotatedTree;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;

import java.util.*;


public class CoreNLPExample {
	
	public int sentimentAnalyzer(String s) {
		
		
		//create pipeline for analysis
		 Properties props = new Properties();
	     props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
	     StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	     int sentScore = 0;
	     
	     
	     if(s != null && s.length() > 0) {
	    	 int len = 0;
	    	 
	    	 //Preprocessing
	    	 Annotation annotation = pipeline.process(s);
	    	 
	    	 //Sentiment Analysis
	    	 for(CoreMap sentence: annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
	    		 Tree tree = sentence.get(SentimentAnnotatedTree.class);
	             int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
	             String partText = sentence.toString();
	             if (partText.length() > len) {
	            	 sentScore = sentiment;
	            	 len = partText.length();
	             }
	    	 }
	     }
	     
	     if (sentScore == 2 || sentScore > 4 || sentScore < 0) {
	            return 0;
	     }
	     
	     return sentScore;
	     
	     

	}
	
	
	
	
	public static void main(String[] args) {
		
		String test = "This is a really positive sample string. "
				+ "Just writing random happy things to test the sentiment score. "
				+ "Been having a great day. Let's test this, because I am happy with the code now!";
		
		CoreNLPExample nlp = new CoreNLPExample();
		int res = nlp.sentimentAnalyzer(test);
		System.out.println(res);
		
		
	}

}
