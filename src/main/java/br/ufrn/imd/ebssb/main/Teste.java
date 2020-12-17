package br.ufrn.imd.ebssb.main;

import java.util.ArrayList;

import br.ufrn.imd.ebssb.core.Dataset;
import br.ufrn.imd.ebssb.core.EbSsBoost;

public class Teste {

	public static ArrayList<Dataset> datasets;
	public static int numFolds = 10;
	public static ArrayList<Dataset> folds;
	public static int seed;

	public static void main(String[] args) throws Exception {
		datasets = new ArrayList<Dataset>();
		folds = new ArrayList<Dataset>();
		populateDatasetsTest();

		seed = 19;

		for (Dataset d : datasets) {
			run(d);
		}
	}

	public static void run(Dataset dataset) throws Exception {

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
			break;
		}
	}

	public static void populateDatasetsTest() {
		String basePath = new String("src/main/resources/datasets/experiment_test/");

		ArrayList<String> sources = new ArrayList<String>();
		sources.add("Iris.arff");

		for (String s : sources) {
			Dataset d;
			try {
				d = new Dataset(basePath + s);
				datasets.add(d);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

}
