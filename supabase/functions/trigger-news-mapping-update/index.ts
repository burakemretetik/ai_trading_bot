
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

// Read environment variables
const supabaseUrl = Deno.env.get("SUPABASE_URL") || "";
const supabaseKey = Deno.env.get("SUPABASE_ANON_KEY") || "";

serve(async (_req) => {
  try {
    // Create a Supabase client
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    // Call the update-stock-news-mapping function
    const response = await fetch(
      `${supabaseUrl}/functions/v1/update-stock-news-mapping`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${supabaseKey}`
        }
      }
    );
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to update stock news mapping: ${response.status} ${errorText}`);
    }
    
    const result = await response.json();
    
    // If the mapping was updated (new news available), trigger email sending
    if (result.updated) {
      const emailResponse = await fetch(
        `${supabaseUrl}/functions/v1/send-stock-news-emails`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${supabaseKey}`
          }
        }
      );
      
      if (!emailResponse.ok) {
        const emailErrorText = await emailResponse.text();
        console.error(`Failed to send email notifications: ${emailResponse.status} ${emailErrorText}`);
      } else {
        const emailResult = await emailResponse.json();
        console.log("Email sending results:", emailResult);
      }
    }
    
    return new Response(
      JSON.stringify({ 
        success: true, 
        mappingUpdated: result.updated,
        timestamp: result.timestamp
      }),
      { 
        status: 200,
        headers: { "Content-Type": "application/json" } 
      }
    );
  } catch (error) {
    console.error("Error in trigger-news-mapping-update function:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
});
