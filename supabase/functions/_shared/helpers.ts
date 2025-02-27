
// Helper functions for Supabase Edge Functions

export function generateMockNews(stockSymbol: string) {
  const sources = ['Bloomberg', 'Reuters', 'Financial Times', 'Wall Street Journal', 'CNBC'];
  const newsCount = Math.floor(Math.random() * 5) + 1; // 1-5 news items
  
  const newsItems = [];
  
  for (let i = 0; i < newsCount; i++) {
    // Generate a date within the last 24 hours
    const date = new Date();
    date.setHours(date.getHours() - Math.floor(Math.random() * 24));
    
    const source = sources[Math.floor(Math.random() * sources.length)];
    const id = crypto.randomUUID();
    
    newsItems.push({
      id,
      title: `${stockSymbol} announces ${Math.random() > 0.5 ? 'positive' : 'negative'} quarterly results`,
      source,
      url: `https://example.com/news/${id}`,
      publishedAt: date.toISOString(),
      summary: `${stockSymbol} reported ${Math.random() > 0.5 ? 'better' : 'worse'} than expected earnings for Q${Math.floor(Math.random() * 4) + 1}. Analysts are ${Math.random() > 0.5 ? 'optimistic' : 'cautious'} about the company's future prospects.`,
    });
  }
  
  return newsItems;
}
