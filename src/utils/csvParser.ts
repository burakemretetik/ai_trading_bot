
/**
 * Utility function to parse CSV data from the BIST stocks file
 */
export const parseCSV = async (filePath: string) => {
  try {
    console.log(`Attempting to fetch CSV from: ${filePath}`);
    const response = await fetch(filePath);
    
    if (!response.ok) {
      console.error(`Failed to fetch CSV: ${response.status} ${response.statusText}`);
      return [];
    }
    
    const csvText = await response.text();
    
    // Split the CSV text into lines
    const lines = csvText.split('\n');
    
    // Remove the header line
    const dataLines = lines.slice(1);
    
    // Filter out empty lines and parse each line into an object
    const parsedData = dataLines
      .filter(line => line.trim() !== '') // Skip empty lines
      .map(line => {
        const [symbol, name, city] = line.split(',');
        return { symbol, name, city };
      });
    
    console.log(`Successfully parsed ${parsedData.length} stocks from CSV`);
    return parsedData;
  } catch (error) {
    console.error('Error parsing CSV:', error);
    return [];
  }
};
