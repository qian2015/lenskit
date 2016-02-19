package org.lenskit.mf.svdfeature;

import org.junit.Test;

import java.io.*;

import static org.junit.Assert.assertThat;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SVDFeatureModelPredictionTest {
    @Test
    public void predictionTest() throws IOException, ClassNotFoundException {
        String testFile = "/opt/pyml/NoisyNegativeImplicitFeedback/data/cand11-clkrat.svdfea";
        //String predFile = "/opt/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-clkrat-exacthmm.pred";
        //String predFile = "/opt/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-clkrat-estexact.pred";
        String predFile = "/opt/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-clkrat-estmerge.pred";
        //String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.exact.hmm.model";
        //String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.est.exact.hmm.model";
        String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.est.merge.hmm.model";

        ObjectInputStream fin = new ObjectInputStream(new FileInputStream(modelFile));
        SVDFeatureModel model = (SVDFeatureModel)(fin.readObject());
        fin.close();

        SVDFeatureInstanceDAO teDao = new SVDFeatureInstanceDAO(new File(testFile), " ");
        BufferedWriter fout = new BufferedWriter(new FileWriter(predFile));
        SVDFeatureInstance ins = teDao.getNextInstance();
        while (ins != null) {
            double prob = model.predict(ins, true);
            fout.write(Double.toString(prob) + "\n");
            ins = teDao.getNextInstance();
        }
        fout.close();
    }
}
