
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";
import { StockNewsMapping } from "../_shared/types.ts";

// CORS headers
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Read environment variables
const supabaseUrl = Deno.env.get("SUPABASE_URL") || "";
const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";

// Create a Supabase client with service role key for admin access
const supabase = createClient(supabaseUrl, supabaseKey);

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }
  
  try {
    // Get all stock news from the last hour
    const oneHourAgo = new Date();
    oneHourAgo.setHours(oneHourAgo.getHours() - 1);
    
    const { data: recentNews, error: newsError } = await supabase
      .from("stock_news")
      .select("stock_symbol, url")
      .gte("created_at", oneHourAgo.toISOString());
    
    if (newsError) {
      throw new Error(`Error fetching recent news: ${newsError.message}`);
    }
    
    // Organize the news by stock symbol
    const stockNews: Record<string, string[]> = {};
    
    for (const news of recentNews || []) {
      if (!stockNews[news.stock_symbol]) {
        stockNews[news.stock_symbol] = [];
      }
      stockNews[news.stock_symbol].push(news.url);
    }
    
    // Create the mapping object
    const mapping: StockNewsMapping = {
      timestamp: new Date().toISOString(),
      updated: Object.keys(stockNews).length > 0,
      stock_news: stockNews
    };
    
    // Upload the mapping file to Supabase Storage
    // First, check if the bucket exists
    const { data: buckets } = await supabase.storage.listBuckets();
    const stockDataBucket = buckets?.find(b => b.name === "stock-data");
    
    if (!stockDataBucket) {
      // Create the bucket if it doesn't exist
      await supabase.storage.createBucket("stock-data", {
        public: true,
        fileSizeLimit: 1024 * 1024 // 1MB limit
      });
    }
    
    // Upload the mapping file
    const { error: uploadError } = await supabase.storage
      .from("stock-data")
      .upload("stock_news_mapping.json", JSON.stringify(mapping), {
        contentType: "application/json",
        upsert: true
      });
    
    if (uploadError) {
      throw new Error(`Error uploading mapping file: ${uploadError.message}`);
    }
    
    return new Response(
      JSON.stringify({ 
        success: true, 
        updated: mapping.updated,
        stocksWithNews: Object.keys(stockNews).length,
        timestamp: mapping.timestamp
      }),
      { 
        status: 200,
        headers: { "Content-Type": "application/json", ...corsHeaders } 
      }
    );
  } catch (error) {
    console.error("Error in update-stock-news-mapping function:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    );
  }
});
