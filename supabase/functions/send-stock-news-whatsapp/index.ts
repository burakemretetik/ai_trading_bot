
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
    const { content, stockNews, recipientPhoneNumber } = await req.json();
    
    console.log("WhatsApp notification request received");
    console.log("Content:", content);
    console.log("Stock news data:", JSON.stringify(stockNews, null, 2));
    console.log("Recipient Phone Number:", recipientPhoneNumber);
    
    // WhatsApp Business API integration
    const whatsappToken = Deno.env.get("WHATSAPP_BUSINESS_TOKEN");
    const whatsappPhoneNumberId = Deno.env.get("WHATSAPP_PHONE_NUMBER_ID");
    
    if (!whatsappToken || !whatsappPhoneNumberId || !recipientPhoneNumber) {
      const missingParams = [];
      if (!whatsappToken) missingParams.push("WHATSAPP_BUSINESS_TOKEN");
      if (!whatsappPhoneNumberId) missingParams.push("WHATSAPP_PHONE_NUMBER_ID");
      if (!recipientPhoneNumber) missingParams.push("recipientPhoneNumber");
      
      console.error(`Missing required parameters: ${missingParams.join(", ")}`);
      return new Response(
        JSON.stringify({ 
          success: false, 
          message: `Missing required parameters: ${missingParams.join(", ")}` 
        }),
        {
          status: 400,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders
          }
        }
      );
    }
    
    try {
      // WhatsApp Business API endpoint
      const url = `https://graph.facebook.com/v17.0/${whatsappPhoneNumberId}/messages`;
      
      // Prepare the request payload
      const payload = {
        messaging_product: "whatsapp",
        recipient_type: "individual",
        to: recipientPhoneNumber,
        type: "text",
        text: { 
          body: content
        }
      };
      
      console.log("Sending WhatsApp message with payload:", JSON.stringify(payload, null, 2));
      
      // Send the message via WhatsApp Business API
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${whatsappToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      
      const result = await response.json();
      console.log("WhatsApp Business API response:", result);
      
      if (response.ok) {
        return new Response(
          JSON.stringify({ 
            success: true, 
            message: "WhatsApp notification sent successfully",
            messageId: result.messages?.[0]?.id
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
        console.error("Error from WhatsApp Business API:", result);
        return new Response(
          JSON.stringify({ 
            success: false, 
            message: "Error from WhatsApp Business API",
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
    } catch (whatsappError) {
      console.error("Error calling WhatsApp Business API:", whatsappError);
      return new Response(
        JSON.stringify({ 
          success: false, 
          message: "Error calling WhatsApp Business API",
          error: whatsappError.message 
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
