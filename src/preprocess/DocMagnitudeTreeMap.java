package preprocess;

import java.util.Set;
import java.util.TreeMap;

/**
 * Created by nuplavikar on 3/3/16.
 */
public class DocMagnitudeTreeMap extends TreeMap<String, Double>
{

	public DocMagnitudeTreeMap(DocMagnitudeTreeMap docMagnitudeTreeMap)
	{
		super(docMagnitudeTreeMap);
	}

	public DocMagnitudeTreeMap()
	{
		super();
	}

	public Double put(String key, Double value)
	{
		return super.put(key, value);
	}

	public Set<String> getSetOfTerms()
	{
		return this.keySet();
	}
	//No need to have a new method which just calls the super's get method.
}
