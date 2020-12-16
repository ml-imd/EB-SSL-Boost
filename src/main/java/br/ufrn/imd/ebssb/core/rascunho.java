private Possibility drawOneChildFromPossibility(Possibility possibility) {
		Possibility drawed = new Possibility(possibility.getKey());
		int aux = pbilRandom.nextInt((int) possibility.getTotalWeight());
		for(Possibility p: possibility.getPossibilities()) {
			aux -= p.getWeight();
			if(aux < 0) {
				drawed.setDrawnValue(p.getKey());
				break;
			}
		}
		return drawed;
	}

	private Possibility drawOneMethod(Possibility possibility) {
		int aux = pbilRandom.nextInt((int) possibility.getTotalWeight());
		Possibility drawed = new Possibility();
		for(Possibility p: possibility.getPossibilities()) {
			aux -= p.getWeight();
			if(aux < 0) {
				drawed = p;
				drawed.setDrawnValue(p.getKey());
				break;
			}
		}
		return drawed;
	}