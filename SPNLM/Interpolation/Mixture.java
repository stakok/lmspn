
import java.io.*;
import java.util.*;

public class Mixture{
	static ArrayList<Double> prob_set_1 = new ArrayList<Double>();
	static ArrayList<Double> prob_set_2 = new ArrayList<Double>();

	public static void main(String[] args) throws Exception{
		String str;
		BufferedReader br1 = new BufferedReader(new InputStreamReader(new FileInputStream("SPN.prob")));
		
		boolean flag = false;
		while( (str = br1.readLine())!=null){
			if(str.equals("end...")) flag = false;
			if(flag) prob_set_1.add(Double.parseDouble(str));
			if(str.equals("start...")) flag = true;
		}

		br1.close();

		BufferedReader br2 = new BufferedReader(new InputStreamReader(new FileInputStream("KN5.prob")));
		while( (str = br2.readLine())!=null){
			prob_set_2.add(Double.parseDouble(str));
		}
		br2.close();

		double prob1 = 0, prob2 = 0, mixture_prob = 0;

		int shift = Math.abs(prob_set_1.size() - prob_set_2.size());
		int size = Math.min(prob_set_1.size(), prob_set_2.size());

		for(int i=0;i<size;i++){
			if(prob_set_1.size()<prob_set_2.size()){
				prob1 += Math.log(prob_set_1.get(i));
				prob2 += Math.log(prob_set_2.get(i+shift));
				mixture_prob += Math.log((prob_set_1.get(i)+prob_set_2.get(i+shift))/2);
			}else{
				prob1 += Math.log(prob_set_1.get(i+shift));
				prob2 += Math.log(prob_set_2.get(i));
				mixture_prob += Math.log((prob_set_1.get(i+shift)+prob_set_2.get(i))/2);
			}
		}

		System.out.println("Perplexity of mixture: " + Math.exp(-mixture_prob/size));
	}
}

