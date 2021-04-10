package br.ufrn.imd.ebssb.main;

import java.util.ArrayList;

import br.ufrn.imd.ebssb.core.Dataset;
import br.ufrn.imd.ebssb.core.EbSsBoost;
import br.ufrn.imd.ebssb.filemanipulation.EbSsBoostOutputWriter;
import br.ufrn.imd.ebssb.results.EbSsBoostResult;
import br.ufrn.imd.ebssb.results.FoldResult;

public class EbSsBoostExecutor {

	public static ArrayList<Dataset> datasets;
	public static int numFolds = 10;
	public static ArrayList<Dataset> folds;
	public static int seed;

	public static EbSsBoostResult ebSsBoostResult;

	public static String ebSsBoostVersionOne = "EbSsB_V_01";

	public static EbSsBoostOutputWriter ebssbowSummary;
	public static String outputSummaryResultBasePath = "src/main/resources/results/summary/";

	public static EbSsBoostOutputWriter ebssbowDetailed;
	public static String outputDetailedResultBasePath = "src/main/resources/results/detailed/";

	public static void main(String[] args) throws Exception {

		datasets = new ArrayList<Dataset>();
		folds = new ArrayList<Dataset>();
		seed = 19;

		populateDatasetsTest();

		for (Dataset d : datasets) {
			run(d, ebSsBoostVersionOne);
		}
	}

	public static void run(Dataset dataset, String ebSsBoostVersion) throws Exception {

		System.out.println("Init EbSsBoost over " + dataset.getDatasetName() + " dataset");
		
		ebSsBoostResult = new EbSsBoostResult(numFolds, dataset.getDatasetName(), ebSsBoostVersion);

		ebssbowSummary = new EbSsBoostOutputWriter(
				outputSummaryResultBasePath + ebSsBoostVersion + "_" + dataset.getDatasetName());

		ebssbowDetailed = new EbSsBoostOutputWriter(
				outputDetailedResultBasePath + ebSsBoostVersion + "_" + dataset.getDatasetName());
		ebssbowDetailed.logGeneralDetails(dataset.getDatasetName());
		ebssbowDetailed.addContentLine(FoldResult.getTableInfoMeans());

		dataset.shuffleInstances(seed);

		folds = Dataset.splitDataset(dataset, numFolds);

		Dataset validation = new Dataset();

		ebSsBoostResult.setBegin(System.currentTimeMillis());

		for (int i = 0; i < numFolds; i++) {

			validation = new Dataset(folds.get(i));

			ArrayList<Dataset> foldsForTest = new ArrayList<Dataset>();
			for (int j = 0; j < numFolds; j++) {
				if (i != j) {
					foldsForTest.add(folds.get(j));

				}
			}

			Dataset ddd = Dataset.joinDatasets(foldsForTest);

			System.out.print("\t fold: " + (i+1) + " ");
			EbSsBoost ebSsBoost = new EbSsBoost(ddd, validation, seed);
			ebSsBoost.runEbSsBoost();

			ebSsBoostResult.setEnd(System.currentTimeMillis());
			ebSsBoostResult.addFoldResult(ebSsBoost.getFoldResult());

			ebssbowDetailed.logDetailsAboutStep(dataset.getDatasetName(), i + 1);
			ebssbowDetailed.addContentLine(ebSsBoost.getFoldResult().foldResultSummarytable());
			// ebssbowDetailed.addContent(ebSsBoost.getFoldResult().getIterationsDatasetHistory());

		}
		ebssbowDetailed.writeInFile();
		
		ebssbowSummary.logGeneralDetails(dataset.getDatasetName());
		ebssbowSummary.addContentLine(ebSsBoostResult.getResult());
		ebssbowSummary.printContent();
		ebssbowSummary.writeInFile();
		
		ebssbowDetailed.saveAndClose();
		ebssbowSummary.saveAndClose();
		System.out.println();
	}

	public static void populateDatasetsTest() {
		String basePath = new String("src/main/resources/datasets/experiment_test/");

		ArrayList<String> sources = new ArrayList<String>();
		//sources.add("Iris.arff");
		sources.add("Abalone.arff");

		for (String s : sources) {
			Dataset d;
			try {
				d = new Dataset(basePath + s);
				datasets.add(d);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public static void populateDatasets() {
		String basePath = new String("src/main/resources/datasets/experiment_all/");

		ArrayList<String> sources = new ArrayList<String>();
		sources.add("Abalone.arff");
		sources.add("Adult.arff");
		sources.add("Arrhythmia.arff");
		sources.add("Automobile.arff");
		sources.add("Btsc.arff");
		sources.add("Car.arff");
		sources.add("Cnae.arff");
		sources.add("Dermatology.arff");
		sources.add("Ecoli.arff");
		sources.add("Flags.arff");
		sources.add("GermanCredit.arff");
		sources.add("Glass.arff");
		sources.add("Haberman.arff");
		sources.add("HillValley.arff");
		sources.add("Ilpd.arff");
		sources.add("ImageSegmentation_norm.arff");
		sources.add("KrVsKp.arff");
		sources.add("Leukemia.arff");
		sources.add("Madelon.arff");
		sources.add("MammographicMass.arff");
		sources.add("MultipleFeaturesKarhunen.arff");
		sources.add("Mushroom.arff");
		sources.add("Musk.arff");
		sources.add("Nursery.arff");
		sources.add("OzoneLevelDetection.arff");
		sources.add("PenDigits.arff");
		sources.add("PhishingWebsite.arff");
		sources.add("Pima.arff");
		sources.add("PlanningRelax.arff");
		sources.add("Secom.arff");
		sources.add("Seeds.arff");
		sources.add("Semeion.arff");
		sources.add("SolarFlare.arff");
		sources.add("SolarFlare1.arff");
		sources.add("Sonar.arff");
		sources.add("SpectfHeart.arff");
		sources.add("TicTacToeEndgame.arff");
		sources.add("Twonorm.arff");
		sources.add("Vehicle.arff");
		sources.add("Waveform.arff");
		sources.add("Wilt.arff");
		sources.add("Wine.arff");
		sources.add("Yeast.arff");

		for (String s : sources) {
			Dataset d;
			try {
				d = new Dataset(basePath + s);
				datasets.add(d);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public static void runTest(Dataset dataset) throws Exception {

		dataset.shuffleInstances(seed);
		folds = Dataset.splitDataset(dataset, numFolds);
		Dataset validation = new Dataset();

		for (int i = 0; i < numFolds; i++) {

			validation = new Dataset(folds.get(i));
			ArrayList<Dataset> foldsForTest = new ArrayList<Dataset>();
			for (int j = 0; j < numFolds; j++) {
				if (i != j) {
					foldsForTest.add(folds.get(j));
				}
			}

			EbSsBoost ssBoost = new EbSsBoost(Dataset.joinDatasets(foldsForTest), validation, seed);
			ssBoost.runEbSsBoost();
			// break;
		}
	}

}
