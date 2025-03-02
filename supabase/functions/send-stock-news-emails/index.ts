
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";
import { EmailSettings, StockNewsMapping } from "../_shared/types.ts";

// CORS headers
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Read environment variables
const supabaseUrl = Deno.env.get("SUPABASE_URL") || "";
const supabaseKey = Deno.env.get("SUPABASE_ANON_KEY") || "";

// Create a Supabase client
const supabase = createClient(supabaseUrl, supabaseKey);

// Function to read the stock news mapping file
async function getStockNewsMapping(): Promise<StockNewsMapping | null> {
  try {
    const response = await fetch(
      `${supabaseUrl}/storage/v1/object/public/stock-data/stock_news_mapping.json`
    );
    
    if (!response.ok) {
      console.error("Failed to fetch stock news mapping:", response.status);
      return null;
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error fetching stock news mapping:", error);
    return null;
  }
}

// Helper function to send emails
async function sendEmailNotifications(
  mapping: StockNewsMapping,
  userEmail: string,
  userStocks: string[]
) {
  // Find news URLs for stocks that the user is tracking
  const relevantNews: Record<string, string[]> = {};
  
  for (const stock of userStocks) {
    if (mapping.stock_news[stock] && mapping.stock_news[stock].length > 0) {
      relevantNews[stock] = mapping.stock_news[stock];
    }
  }
  
  // If there's no relevant news, don't send an email
  if (Object.keys(relevantNews).length === 0) {
    return;
  }
  
  // Format the email content
  let emailContent = `<h1>Stock News Update</h1>
                     <p>Here are the latest news articles for stocks you're tracking:</p>`;
  
  for (const [symbol, urls] of Object.entries(relevantNews)) {
    emailContent += `<h2>${symbol}</h2><ul>`;
    for (const url of urls) {
      emailContent += `<li><a href="${url}">${url}</a></li>`;
    }
    emailContent += `</ul>`;
  }
  
  // Here you would integrate with an email service like Resend
  // This is a placeholder for the actual email sending logic
  console.log(`Would send email to ${userEmail} with content:`, emailContent);
  
  // Return success for now
  return { success: true, email: userEmail };
}

// Main handler
serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }
  
  try {
    const mapping = await getStockNewsMapping();
    
    // If there's no mapping or it's not updated, don't proceed
    if (!mapping || !mapping.updated || Object.keys(mapping.stock_news).length === 0) {
      return new Response(
        JSON.stringify({ message: "No updated news to send" }),
        { 
          status: 200,
          headers: { "Content-Type": "application/json", ...corsHeaders } 
        }
      );
    }
    
    // Get users with email notifications enabled
    const { data: users, error: userError } = await supabase
      .from("profiles")
      .select("id, email_settings")
      .eq("email_settings->enabled", true);
    
    if (userError) {
      throw new Error(`Error fetching users: ${userError.message}`);
    }
    
    // Get tracked stocks for each user
    const emailResults = [];
    
    for (const user of users || []) {
      const { data: stocks, error: stockError } = await supabase
        .from("tracked_stocks")
        .select("symbol")
        .eq("user_id", user.id);
      
      if (stockError) {
        console.error(`Error fetching stocks for user ${user.id}: ${stockError.message}`);
        continue;
      }
      
      const userStocks = stocks?.map(stock => stock.symbol) || [];
      
      if (userStocks.length > 0) {
        const emailSettings = user.email_settings as EmailSettings;
        const result = await sendEmailNotifications(mapping, emailSettings.address, userStocks);
        if (result) {
          emailResults.push(result);
        }
      }
    }
    
    return new Response(
      JSON.stringify({ 
        success: true, 
        emailsSent: emailResults.length,
        results: emailResults 
      }),
      { 
        status: 200,
        headers: { "Content-Type": "application/json", ...corsHeaders } 
      }
    );
  } catch (error) {
    console.error("Error in send-stock-news-emails function:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    );
  }
});
