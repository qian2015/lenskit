package org.lenskit.mf.svdfeature;

import org.junit.Test;
import org.lenskit.solver.objective.LogisticLoss;

import java.io.*;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SVDFeatureModelBuildTest {
    @Test
    public void testModelBuild() throws IOException {
        //String train = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.tr";
        //String train = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.exact.hmm.tr";
        //String train = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.est.exact.hmm.tr";
        String train = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.est.merge.hmm.tr";
        int numBiases = 38544;
        int numFactors = 38543;
        int dim = 20;
        SVDFeatureInstanceDAO dao = new SVDFeatureInstanceDAO(new File(train), " ");
        LogisticLoss loss = new LogisticLoss();
        SVDFeatureModelBuilder modelBuilder = new SVDFeatureModelBuilder(numBiases, numFactors,
                dim, dao, loss);
        SVDFeatureModel model = modelBuilder.build();
        String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.est.merge.hmm.model";
        ObjectOutputStream fout = new ObjectOutputStream(new FileOutputStream(modelFile));
        fout.writeObject(model);
        fout.close();
    }
}
