
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
    
    // Twilio API integration
    const twilio = {
      accountSid: Deno.env.get("TWILIO_ACCOUNT_SID"),
      authToken: Deno.env.get("TWILIO_AUTH_TOKEN"),
      fromNumber: Deno.env.get("TWILIO_WHATSAPP_NUMBER"),
      toNumber: Deno.env.get("RECIPIENT_WHATSAPP_NUMBER"),
    };
    
    if (!twilio.accountSid || !twilio.authToken || !twilio.fromNumber || !twilio.toNumber) {
      console.error("Missing Twilio credentials or phone numbers");
      return new Response(
        JSON.stringify({ 
          success: false, 
          message: "Missing Twilio credentials or phone numbers" 
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
      // Twilio API endpoint for sending WhatsApp messages
      const url = `https://api.twilio.com/2010-04-01/Accounts/${twilio.accountSid}/Messages.json`;
      
      // Format the body content for Twilio WhatsApp
      const formData = new URLSearchParams();
      formData.append('From', `whatsapp:${twilio.fromNumber}`);
      formData.append('To', `whatsapp:${twilio.toNumber}`);
      formData.append('Body', content);
      
      // Create Authorization header for Twilio
      const authHeader = 'Basic ' + btoa(`${twilio.accountSid}:${twilio.authToken}`);
      
      // Send the message via Twilio API
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": authHeader,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: formData,
      });
      
      const result = await response.json();
      console.log("Twilio API response:", result);
      
      if (response.ok) {
        return new Response(
          JSON.stringify({ 
            success: true, 
            message: "WhatsApp notification sent successfully via Twilio",
            messageId: result.sid
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
