package br.ufrn.imd.ebssb.main;

import java.util.ArrayList;

import br.ufrn.imd.ebssb.core.Dataset;
import br.ufrn.imd.ebssb.core.SelfTraining;
import br.ufrn.imd.ebssb.core.SelfTrainingEnsembleBased;
import br.ufrn.imd.ebssb.core.SelfTrainingEnsembleBasedTest;
import br.ufrn.imd.ebssb.core.SelfTrainingStandard;
import br.ufrn.imd.ebssb.filemanipulation.SelfTrainingOutputWriter;
import br.ufrn.imd.ebssb.results.SelfTrainingResult;

public class Main {

	public static ArrayList<Dataset> datasets;
	public static int numFolds = 10;
	public static SelfTraining st;
	public static ArrayList<Dataset> folds;
	public static SelfTrainingResult str;
	public static int seed;

	public static String selfTrainingVersionOne = "ST_VERSION_01";
	public static String selfTrainingVersionTwo = "ST_VERSION_02";
	public static String selfTrainingStandard = "ST_VERSION_STANDARD";
	
	public static String selfTrainingVersionOneDistance = "ST_V_01_DISTANCE";
	public static String selfTrainingVersionTwoDistance = "ST_V_02_DISTANCE";
	public static String selfTrainingStandardDistance = "ST_VERSION_STD_DISTANCE";
	
	public static String selfTrainingTest = "ST_VERSION_TEST";
	
	public static SelfTrainingOutputWriter sow;
	public static String outputResultBasePath = "src/main/resources/results/";
	
	public static void main(String[] args) throws Exception {
		datasets = new ArrayList<Dataset>();
		folds = new ArrayList<Dataset>();
		populateDatasets();

		seed = 19;
		
		for (Dataset d : datasets) {
			
			run(d, selfTrainingTest);
		}
	}

	public static void populateDatasets() {
		String basePath = new String("src/main/resources/datasets/experiment_all/");
		
		

		
		
		ArrayList<String> sources = new ArrayList<String>();
		//sources.add("Abalone.arff");
		//sources.add("Adult.arff");
		sources.add("Arrhythmia.arff");
		sources.add("Automobile.arff");
		sources.add("Btsc.arff");
		//sources.add("Car.arff");
		//sources.add("Cnae.arff");
		sources.add("Dermatology.arff");
		sources.add("Ecoli.arff");
		sources.add("Flags.arff");
		sources.add("GermanCredit.arff");
		sources.add("Glass.arff");
		sources.add("Haberman.arff");
		sources.add("HillValley.arff");
		sources.add("Ilpd.arff");
		//sources.add("ImageSegmentation_norm.arff");
		//sources.add("KrVsKp.arff"); 
		sources.add("Leukemia.arff");
		//sources.add("Madelon.arff"); 
		sources.add("MammographicMass.arff");
		//sources.add("MultipleFeaturesKarhunen.arff"); 
		//sources.add("Mushroom.arff");
		//sources.add("Musk.arff"); 
		//sources.add("Nursery.arff");
		//sources.add("OzoneLevelDetection.arff"); 
		//sources.add("PenDigits.arff");
		//sources.add("PhishingWebsite.arff"); 
		sources.add("Pima.arff");
		sources.add("PlanningRelax.arff"); 
		//sources.add("Secom.arff");
		sources.add("Seeds.arff"); 
		//sources.add("Semeion.arff");
		//sources.add("SolarFlare.arff"); 
		sources.add("SolarFlare1.arff");
		sources.add("Sonar.arff"); 
		sources.add("SpectfHeart.arff");
		sources.add("TicTacToeEndgame.arff"); 
		//sources.add("Twonorm.arff");
		sources.add("Vehicle.arff"); 
		//sources.add("Waveform.arff");
		//sources.add("Wilt.arff"); 
		//sources.add("Wine.arff");
		//sources.add("Yeast.arff");

		for(String s: sources) {
			Dataset d;
			try {
				d = new Dataset(basePath+s);
				datasets.add(d);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public static void run(Dataset dataset, String selfTrainingVersion) throws Exception {
		
		str = new SelfTrainingResult(numFolds, dataset.getDatasetName(), selfTrainingVersion);
		sow = new SelfTrainingOutputWriter(outputResultBasePath + selfTrainingVersion + "_ERROR_" + dataset.getDatasetName());

		dataset.shuffleInstances(seed);
		folds = Dataset.splitDataset(dataset, numFolds);
		Dataset validation = new Dataset();
		str.setBegin(System.currentTimeMillis());
		
		for (int i = 0; i < numFolds; i++) {
			
			validation = new Dataset(folds.get(i));
			ArrayList<Dataset> foldsForTest = new ArrayList<Dataset>();
			for (int j = 0; j < numFolds; j++) {
				if (i != j) {
					foldsForTest.add(folds.get(j));
				}
			}
			
			if(selfTrainingVersion.equals(selfTrainingStandard)) {
				SelfTrainingStandard sts = new SelfTrainingStandard(Dataset.joinDatasets(foldsForTest), validation);
				sts.runStandard();
				st = (SelfTrainingStandard)sts;
			}
			else if(selfTrainingVersion.equals(selfTrainingVersionOne)) {
				SelfTrainingEnsembleBased steb = new SelfTrainingEnsembleBased(Dataset.joinDatasets(foldsForTest), validation);
				steb.runVersionOne();
				st = (SelfTrainingEnsembleBased)steb;
			}
			else if(selfTrainingVersion.equals(selfTrainingVersionTwo)) {
				SelfTrainingEnsembleBased steb = new SelfTrainingEnsembleBased(Dataset.joinDatasets(foldsForTest), validation);
				steb.runVersionTwo();
				st = (SelfTrainingEnsembleBased)steb;
			}
			else if(selfTrainingVersion.equals(selfTrainingStandardDistance)) {
				SelfTrainingStandard sts = new SelfTrainingStandard(Dataset.joinDatasets(foldsForTest), validation);
				sts.runStandardDistanceFactor();
				st = (SelfTrainingStandard)sts;
			}
			else if(selfTrainingVersion.equals(selfTrainingVersionOneDistance)) {
				SelfTrainingEnsembleBased steb = new SelfTrainingEnsembleBased(Dataset.joinDatasets(foldsForTest), validation);
				steb.runVersionOneDistanceFactor();
				st = (SelfTrainingEnsembleBased)steb;
			}
			else if(selfTrainingVersion.equals(selfTrainingVersionTwoDistance)) {
				SelfTrainingEnsembleBased steb = new SelfTrainingEnsembleBased(Dataset.joinDatasets(foldsForTest), validation);
				steb.runVersionTwoDistanceFactor();
				st = (SelfTrainingEnsembleBased)steb;
			}
			else if(selfTrainingVersion.equals(selfTrainingTest)) {
				SelfTrainingEnsembleBasedTest stebt = new SelfTrainingEnsembleBasedTest(Dataset.joinDatasets(foldsForTest), validation);
				stebt.runTest();
				st = (SelfTrainingEnsembleBasedTest)stebt;
			}
		
			str.setEnd(System.currentTimeMillis());
			str.addFoldResult(st.getResult());
			
			sow.logDetailsAboutStep(dataset.getDatasetName(), i);
			sow.addContentline(st.getHistory());
		}
		sow.addContentline(str.getResult());
		sow.writeInFile();
		
		str.showResult();
	}
	

}
