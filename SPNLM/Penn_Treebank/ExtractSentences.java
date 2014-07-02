import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
import java.io.*;
import java.util.*;

public class ExtractSentences {

	static class Pair{
		private String word;
		private int count;

		public Pair(String word, int count){
			this.word = word;
			this.count = count;
		}

		public void setCount(int count){
			this.count = count;
		}
		public String getWord(){
			return word;
		}
		public int getCount(){
			return count;
		}
	}

	static ArrayList<Pair> dict = new ArrayList<Pair>();	//dictionary

	public static void main(String args[]){

		try {
			//read the training corpus
			ArrayList<String> train_corpus = new ArrayList<String>();

			File fXmlFile = new File(args[0]);
			DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
			Document doc = dBuilder.parse(fXmlFile);

			doc.getDocumentElement().normalize();

			NodeList nList = doc.getElementsByTagName("terminals");

			int tkn_num = 0;
			for (int temp = 0; temp < nList.getLength(); temp++) {

				Node nNode = nList.item(temp);

				if (nNode.getNodeType() == Node.ELEMENT_NODE) {

					Element eElement = (Element) nNode;
					NodeList aList = eElement.getElementsByTagName("t");
					tkn_num += aList.getLength();

					for(int i=0;i<aList.getLength();i++){
						train_corpus.add(((Element)aList.item(i)).getAttribute("word"));
					}
				}
				train_corpus.add("</s>");
			}

			//Build the dictionary
			for(int i=0;i<train_corpus.size();i++){
				int idx = -1;
				for(int j=0;j<dict.size();j++){
					if(dict.get(j).getWord().equals(train_corpus.get(i))){
						idx = j;
						break;
					}
				}
				if(idx==-1){
					dict.add(new Pair(train_corpus.get(i), 1));
				}else{
					dict.get(idx).setCount(dict.get(idx).getCount()+1);
				}
			}

			Collections.sort(dict, new Comparator<Pair>(){
				public int compare(Pair p1, Pair p2) {
					return - p1.getCount() + p2.getCount();
				}
			});

			ArrayList<String> top_dict = new ArrayList<String>();
			for(int i=0;i<9999;i++){
				top_dict.add(dict.get(i).getWord());
			}

			PrintWriter train_corpus_idx = new PrintWriter(new OutputStreamWriter(new FileOutputStream(args[3])));
			for(int i=0;i<train_corpus.size();i++){
				int idx = top_dict.indexOf(train_corpus.get(i));
				if(idx==-1){
					train_corpus_idx.print("0 ");
				}else{
					train_corpus_idx.print((idx+1) + " ");
				}
			}
			train_corpus_idx.close();
			


			//read the validation corpus
			ArrayList<String> valid_corpus = new ArrayList<String>();

			fXmlFile = new File(args[1]);
			dbFactory = DocumentBuilderFactory.newInstance();
			dBuilder = dbFactory.newDocumentBuilder();
			doc = dBuilder.parse(fXmlFile);

			doc.getDocumentElement().normalize();

			nList = doc.getElementsByTagName("terminals");

			tkn_num = 0;
			for (int temp = 0; temp < nList.getLength(); temp++) {

				Node nNode = nList.item(temp);

				if (nNode.getNodeType() == Node.ELEMENT_NODE) {

					Element eElement = (Element) nNode;
					NodeList aList = eElement.getElementsByTagName("t");
					tkn_num += aList.getLength();

					for(int i=0;i<aList.getLength();i++){
						valid_corpus.add(((Element)aList.item(i)).getAttribute("word"));
					}
				}
				valid_corpus.add("</s>");
			}

			PrintWriter valid_corpus_idx = new PrintWriter(new OutputStreamWriter(new FileOutputStream(args[4])));
			for(int i=0;i<valid_corpus.size();i++){
				int idx = top_dict.indexOf(valid_corpus.get(i));
				if(idx==-1){
					valid_corpus_idx.print("0 ");
				}else{
					valid_corpus_idx.print((idx+1) + " ");
				}
			}
			valid_corpus_idx.close();
			

			//read the testing corpus
			ArrayList<String> test_corpus = new ArrayList<String>();

			fXmlFile = new File(args[2]);
			dbFactory = DocumentBuilderFactory.newInstance();
			dBuilder = dbFactory.newDocumentBuilder();
			doc = dBuilder.parse(fXmlFile);

			doc.getDocumentElement().normalize();

			nList = doc.getElementsByTagName("terminals");

			tkn_num = 0;
			for (int temp = 0; temp < nList.getLength(); temp++) {

				Node nNode = nList.item(temp);

				if (nNode.getNodeType() == Node.ELEMENT_NODE) {

					Element eElement = (Element) nNode;
					NodeList aList = eElement.getElementsByTagName("t");
					tkn_num += aList.getLength();

					for(int i=0;i<aList.getLength();i++){
						test_corpus.add(((Element)aList.item(i)).getAttribute("word"));
					}
				}
				test_corpus.add("</s>");
			}

			PrintWriter test_corpus_idx = new PrintWriter(new OutputStreamWriter(new FileOutputStream(args[5])));
			for(int i=0;i<test_corpus.size();i++){
				int idx = top_dict.indexOf(test_corpus.get(i));
				if(idx==-1){
					test_corpus_idx.print("0 ");
				}else{
					test_corpus_idx.print((idx+1) + " ");
				}
			}
			test_corpus_idx.close();
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
 
}
