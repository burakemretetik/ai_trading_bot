
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";

// CORS headers
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Handle WhatsApp notification request
serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { content, stockNews } = await req.json();
    
    console.log("WhatsApp notification request received");
    console.log("Content:", content);
    console.log("Stock news data:", JSON.stringify(stockNews, null, 2));
    
    // In a real implementation, you would use the WhatsApp Business API or a service like Twilio
    // Example with Twilio (uncomment and configure with your Twilio credentials):
    
    const twilioAccountSid = Deno.env.get("TWILIO_ACCOUNT_SID");
    const twilioAuthToken = Deno.env.get("TWILIO_AUTH_TOKEN");
    const twilioWhatsAppNumber = Deno.env.get("TWILIO_WHATSAPP_NUMBER");
    const userWhatsAppNumber = Deno.env.get("USER_WHATSAPP_NUMBER"); // The recipient's number
    
    if (!twilioAccountSid || !twilioAuthToken || !twilioWhatsAppNumber || !userWhatsAppNumber) {
      console.error("Missing Twilio credentials or WhatsApp numbers");
      return new Response(
        JSON.stringify({ 
          success: false, 
          message: "Missing Twilio credentials or WhatsApp numbers" 
        }),
        {
          status: 500,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders
          }
        }
      );
    }
    
    try {
      const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${twilioAccountSid}/Messages.json`, {
        method: "POST",
        headers: {
          "Authorization": `Basic ${btoa(`${twilioAccountSid}:${twilioAuthToken}`)}`,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          From: `whatsapp:${twilioWhatsAppNumber}`,
          To: `whatsapp:${userWhatsAppNumber}`,
          Body: content,
        }),
      });
      
      const result = await response.json();
      console.log("Twilio API response:", result);
      
      if (response.ok) {
        return new Response(
          JSON.stringify({ 
            success: true, 
            message: "WhatsApp notification sent successfully",
            sid: result.sid 
          }),
          {
            status: 200,
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders
            }
          }
        );
      } else {
        console.error("Error from Twilio API:", result);
        return new Response(
          JSON.stringify({ 
            success: false, 
            message: "Error from Twilio API",
            error: result 
          }),
          {
            status: 500,
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders
            }
          }
        );
      }
    } catch (twilioError) {
      console.error("Error calling Twilio API:", twilioError);
      return new Response(
        JSON.stringify({ 
          success: false, 
          message: "Error calling Twilio API",
          error: twilioError.message 
        }),
        {
          status: 500,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders
          }
        }
      );
    }
  } catch (error) {
    console.error("Error in send-stock-news-whatsapp function:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      }
    );
  }
});
