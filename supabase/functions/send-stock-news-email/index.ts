
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { Resend } from "npm:resend@2.0.0";

// Configure Resend API
const resend = new Resend(Deno.env.get("RESEND_API_KEY"));

// CORS headers
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Handle email sending request
serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { content, subject, stockNews } = await req.json();
    
    // Here we would normally fetch the user's email from the auth system
    // For now, we'll use a dummy email or retrieve it from request
    // In a real app, you'd get this from the authenticated user
    const userEmail = "user@example.com"; // Replace with actual user email
    
    if (!content || !subject) {
      throw new Error("Missing required email parameters");
    }
    
    // Send the email
    const { data, error } = await resend.emails.send({
      from: "Stock News <onboarding@resend.dev>",
      to: [userEmail],
      subject: subject,
      html: content,
    });
    
    if (error) {
      console.error("Error sending email:", error);
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
    
    console.log("Email sent successfully:", data);
    
    return new Response(
      JSON.stringify({ 
        success: true, 
        message: "Email sent successfully",
        data 
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      }
    );
  } catch (error) {
    console.error("Error in send-stock-news-email function:", error);
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
