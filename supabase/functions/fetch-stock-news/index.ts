
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.38.4'

const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? ''
const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface Stock {
  id: string
  symbol: string
  name: string
  tracked: boolean
}

interface NewsItem {
  id: string
  title: string
  source: string
  url: string
  publishedAt: string
  summary: string
}

Deno.serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey)
    
    // Get all tracked stocks
    const { data: stocks, error: stocksError } = await supabase
      .from('tracked_stocks')
      .select('*')
    
    if (stocksError) {
      console.error('Error fetching tracked stocks:', stocksError)
      
      // If the tracked_stocks table doesn't exist yet, try to get tracked stocks from localStorage
      console.log('Trying to fetch all stocks since tracked_stocks table might not exist yet')
      const { data: allStocks, error: allStocksError } = await supabase
        .rpc('get_tracked_stocks_from_json')
      
      if (allStocksError) {
        console.error('Error fetching all stocks:', allStocksError)
        throw new Error('Failed to fetch stocks')
      }
      
      console.log(`Found ${allStocks?.length || 0} stocks from JSON function`)
      return await processStocks(allStocks, supabase)
    }
    
    console.log(`Found ${stocks?.length || 0} tracked stocks`)
    return await processStocks(stocks, supabase)
    
  } catch (error) {
    console.error('Error in fetch-stock-news function:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      }
    )
  }
})

async function processStocks(stocks: Stock[], supabase: any) {
  if (!stocks || stocks.length === 0) {
    console.log('No stocks to process')
    return new Response(
      JSON.stringify({ message: 'No tracked stocks found' }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )
  }

  const newsPromises = stocks.map(async (stock) => {
    try {
      // Mock Google search API call for now
      // In production, you would use the Google Search API or another news API
      const news = await mockGoogleSearch(stock.symbol)
      
      // Insert news items into the database
      if (news && news.length > 0) {
        const newsItems = news.map(item => ({
          stock_id: stock.id,
          stock_symbol: stock.symbol,
          title: item.title,
          source: item.source,
          url: item.url,
          published_at: item.publishedAt,
          summary: item.summary,
        }))
        
        // Delete old news for this stock (older than 3 days)
        const threeDaysAgo = new Date()
        threeDaysAgo.setDate(threeDaysAgo.getDate() - 3)
        
        const { error: deleteError } = await supabase
          .from('stock_news')
          .delete()
          .eq('stock_symbol', stock.symbol)
          .lt('created_at', threeDaysAgo.toISOString())
        
        if (deleteError) {
          console.error(`Error deleting old news for ${stock.symbol}:`, deleteError)
        }
        
        // Insert new news items
        const { error: insertError } = await supabase
          .from('stock_news')
          .insert(newsItems)
        
        if (insertError) {
          console.error(`Error inserting news for ${stock.symbol}:`, insertError)
        } else {
          console.log(`Successfully inserted ${newsItems.length} news items for ${stock.symbol}`)
        }
      } else {
        console.log(`No news found for ${stock.symbol}`)
      }
      
      return { stock: stock.symbol, newsCount: news?.length || 0 }
    } catch (error) {
      console.error(`Error processing stock ${stock.symbol}:`, error)
      return { stock: stock.symbol, error: error.message }
    }
  })

  const results = await Promise.all(newsPromises)
  
  return new Response(
    JSON.stringify({ 
      message: 'News fetch completed',
      results 
    }),
    {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    }
  )
}

// Mock function to simulate Google search results
async function mockGoogleSearch(stockSymbol: string): Promise<NewsItem[]> {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 500))
  
  // Generate some random news items
  const sources = ['Bloomberg', 'Reuters', 'Financial Times', 'Wall Street Journal', 'CNBC']
  const newsCount = Math.floor(Math.random() * 5) + 1 // 1-5 news items
  
  const newsItems: NewsItem[] = []
  
  for (let i = 0; i < newsCount; i++) {
    // Generate a date within the last 24 hours
    const date = new Date()
    date.setHours(date.getHours() - Math.floor(Math.random() * 24))
    
    const source = sources[Math.floor(Math.random() * sources.length)]
    const id = crypto.randomUUID()
    
    newsItems.push({
      id,
      title: `${stockSymbol} announces ${Math.random() > 0.5 ? 'positive' : 'negative'} quarterly results`,
      source,
      url: `https://example.com/news/${id}`,
      publishedAt: date.toISOString(),
      summary: `${stockSymbol} reported ${Math.random() > 0.5 ? 'better' : 'worse'} than expected earnings for Q${Math.floor(Math.random() * 4) + 1}. Analysts are ${Math.random() > 0.5 ? 'optimistic' : 'cautious'} about the company's future prospects.`,
    })
  }
  
  return newsItems
}
