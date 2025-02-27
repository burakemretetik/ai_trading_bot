
/**
 * Utility function to parse CSV data from the BIST stocks file
 */
export const parseCSV = async (filePath: string) => {
  try {
    const response = await fetch(filePath);
    const csvText = await response.text();
    
    // Split the CSV text into lines
    const lines = csvText.split('\n');
    
    // Remove the header line
    const dataLines = lines.slice(1);
    
    // Parse each line into an object
    return dataLines
      .filter(line => line.trim() !== '') // Skip empty lines
      .map(line => {
        const [symbol, name, city] = line.split(',');
        return { symbol, name, city };
      });
  } catch (error) {
    console.error('Error parsing CSV:', error);
    return [];
  }
};
