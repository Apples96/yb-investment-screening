"""
Workflow Code File

This file contains the workflow code that will be executed when testing workflows.
Copy and paste your generated workflow code here before clicking "Test Workflow".

Instructions:
1. Generate workflow code using the "Create Workflow" section
2. Copy the generated code from the UI
3. Paste it here, replacing this placeholder
4. Click "Test Workflow" to execute the code from this file

The workflow code should be a complete Python script that can be executed independently.
"""

# Paste your workflow code below this line

import asyncio
import aiohttp
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

# Configuration - replace with your actual values
LIGHTON_API_KEY = "your_api_key_here"
LIGHTON_BASE_URL = "https://api.lighton.ai"

logger = logging.getLogger(__name__)


class ParadigmClient:
    def __init__(self, api_key: str, base_url: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def document_search(self, query: str, **kwargs) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/api/v2/chat/document-search"
        payload = {"query": query, **kwargs}

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API error {response.status}: {await response.text()}")

    async def analyze_documents_with_polling(self, query: str, document_ids: List[int], **kwargs) -> str:
        # Start analysis
        endpoint = f"{self.base_url}/api/v2/chat/document-analysis"
        payload = {"query": query, "document_ids": document_ids, **kwargs}

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    chat_response_id = result.get("chat_response_id")
                else:
                    raise Exception(f"Analysis API error {response.status}: {await response.text()}")

        # Poll for results
        max_wait = 300  # 5 minutes
        poll_interval = 5
        elapsed = 0

        while elapsed < max_wait:
            endpoint = f"{self.base_url}/api/v2/chat/document-analysis/{chat_response_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=self.headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        status = result.get("status", "")
                        if status.lower() in ["completed", "complete", "finished", "success"]:
                            analysis_result = result.get("result") or result.get("detailed_analysis") or "Analysis completed"
                            return analysis_result
                        elif status.lower() in ["failed", "error"]:
                            raise Exception(f"Analysis failed: {status}")
                    elif response.status == 404:
                        # Analysis not ready yet, continue polling
                        pass
                    else:
                        raise Exception(f"Polling API error {response.status}: {await response.text()}")

                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

        raise Exception("Analysis timed out")

    async def chat_completion(self, prompt: str, model: str = "alfred-4.2") -> str:
        endpoint = f"{self.base_url}/api/v2/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"Paradigm chat completion API error {response.status}: {await response.text()}")

    async def analyze_image(self, query: str, document_ids: List[str], model: str = None, private: bool = False) -> str:
        endpoint = f"{self.base_url}/api/v2/chat/image-analysis"
        payload = {
            "query": query,
            "document_ids": document_ids
        }
        if model:
            payload["model"] = model
        if private is not None:
            payload["private"] = private

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("answer", "No analysis result provided")
                else:
                    raise Exception(f"Image analysis API error {response.status}: {await response.text()}")


# Initialize clients
paradigm_client = ParadigmClient(LIGHTON_API_KEY, LIGHTON_BASE_URL)


async def execute_workflow(user_input: str) -> str:
    # STEP 1: Receive and identify the investment opportunity document
    attached_file_ids = globals().get('attached_file_ids', [])

    if not attached_file_ids:
        return "Error: No investment opportunity document provided. Please attach the document to analyze."

    document_reference = [str(file_id) for file_id in attached_file_ids]

    # STEP 2: Search for and retrieve the investment opportunity document
    search_kwargs = {"query": "investment opportunity document type email pitch deck", "file_ids": attached_file_ids}
    search_result = await paradigm_client.document_search(**search_kwargs)

    documents = search_result.get("documents", [])
    if not documents:
        return "Error: Could not retrieve the investment opportunity document."

    document_ids = [str(doc["id"]) for doc in documents]
    document_metadata = {
        "ids": document_ids,
        "titles": [doc.get("title", "Unknown") for doc in documents],
        "types": [doc.get("content_type", "Unknown") for doc in documents]
    }

    # STEP 3: Analyze the investment opportunity document
    analysis_query = """Please provide a comprehensive analysis of this investment opportunity document. Extract the following key information:

1. Target company name and full legal entity structure
2. Detailed business model description including products/services offered
3. Geographic presence and expansion plans, specifically mentioning any GCC region intentions
4. Financial information including current EBITDA status, runway to profitability, funding requirements, and dividend policy if mentioned
5. Investment terms including proposed ticket size, management fees structure, and timeline expectations
6. Sector classification and sub-sector details
7. Return projections including IRR if provided
8. Investor syndicate composition including lead investor status
9. Partnership or joint venture structures if applicable
10. Any mentions of KGI involvement or co-investment opportunities

Please be thorough and specific in your analysis, noting when information is not available."""

    if len(document_ids) > 5:
        # Process in batches of 5
        analysis_results = []
        for i in range(0, len(document_ids), 5):
            batch = document_ids[i:i+5]
            result = await paradigm_client.analyze_documents_with_polling(analysis_query, batch)
            analysis_results.append(result)
        detailed_analysis = "\n\n".join(analysis_results)
    else:
        detailed_analysis = await paradigm_client.analyze_documents_with_polling(analysis_query, document_ids)

    # Initialize evaluation results
    criteria_evaluations = {}

    # STEP 4: Evaluate Criterion 1 (Geography/Structure)
    geo_evaluation = {"status": "NOT MET", "explanation": "", "color": "🔴"}

    # Check for GCC JV opportunity
    gcc_jv_found = False
    if "gcc" in detailed_analysis.lower() and ("joint venture" in detailed_analysis.lower() or "jv" in detailed_analysis.lower()):
        if any(keyword in detailed_analysis.lower() for keyword in ["expansion", "partner", "business model", "proven"]):
            gcc_jv_found = True

    # Check for dividend-paying investment
    dividend_found = False
    if "dividend" in detailed_analysis.lower():
        # Look for yield percentage
        import re
        yield_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', detailed_analysis)
        for match in yield_matches:
            if float(match) > 4:
                dividend_found = True
                break

    # Check for KGI co-investment
    kgi_found = "kgi" in detailed_analysis.lower() and ("co-investment" in detailed_analysis.lower() or "participation" in detailed_analysis.lower())

    if gcc_jv_found:
        geo_evaluation = {"status": "MET", "explanation": "GCC JV opportunity identified with expansion plans and partner structure", "color": "🟢"}
    elif dividend_found:
        geo_evaluation = {"status": "MET", "explanation": "Dividend-paying investment with yield greater than 4%", "color": "🟢"}
    elif kgi_found:
        geo_evaluation = {"status": "MET", "explanation": "KGI co-investment opportunity identified", "color": "🟢"}
    else:
        geo_evaluation = {"status": "NOT MET", "explanation": "Does not meet any of the three required categories: GCC JV, dividend-paying (>4%), or KGI co-investment", "color": "🔴"}

    criteria_evaluations["Geography/Structure"] = geo_evaluation

    # STEP 5: Evaluate Criterion 2 (Financial Milestones)
    financial_evaluation = {"status": "NOT MET", "explanation": "", "color": "🔴"}

    # Check if it's a new JV
    is_new_jv = "new" in detailed_analysis.lower() and ("joint venture" in detailed_analysis.lower() or "jv" in detailed_analysis.lower())

    if is_new_jv:
        financial_evaluation = {"status": "MET", "explanation": "Not applicable - New JV", "color": "🟢"}
    else:
        # Check EBITDA status
        ebitda_positive = "ebitda positive" in detailed_analysis.lower() or "positive ebitda" in detailed_analysis.lower()

        if ebitda_positive:
            financial_evaluation = {"status": "MET", "explanation": "Company is already EBITDA positive", "color": "🟢"}
        else:
            # Look for timeline information
            timeline_within_year = False
            if any(phrase in detailed_analysis.lower() for phrase in ["within one year", "12 months", "less than a year"]):
                timeline_within_year = True

            additional_funding_needed = any(phrase in detailed_analysis.lower() for phrase in ["additional funding", "more funding", "next round", "series"])

            if timeline_within_year and not additional_funding_needed:
                financial_evaluation = {"status": "MET", "explanation": "Timeline to positive EBITDA is within one year with current funding", "color": "🟢"}
            else:
                financial_evaluation = {"status": "NOT MET", "explanation": "Timeline exceeds one year or additional funding rounds needed before profitability", "color": "🔴"}

    criteria_evaluations["Financial Milestones"] = financial_evaluation

    # STEP 6: Evaluate Criterion 3 (Asset Class Exclusion)
    asset_class_evaluation = {"status": "NOT MET", "explanation": "", "color": "🔴"}

    is_fund = any(fund_type in detailed_analysis.lower() for fund_type in ["venture fund", "pe fund", "hedge fund", "fund investment", "pooled investment"])

    if is_fund:
        asset_class_evaluation = {"status": "NOT MET", "explanation": "Fund investment identified - excluded due to team bandwidth and 2025 objectives", "color": "🔴"}
    else:
        # Check if it's clearly a direct company investment
        is_direct = any(phrase in detailed_analysis.lower() for phrase in ["company", "business", "startup", "direct investment"])
        if is_direct:
            asset_class_evaluation = {"status": "MET", "explanation": "Direct company investment identified", "color": "🟢"}
        else:
            asset_class_evaluation = {"status": "NOT MET", "explanation": "Asset class information unclear or absent", "color": "🔴"}

    criteria_evaluations["Asset Class Exclusion"] = asset_class_evaluation

    # STEP 7: Evaluate Criterion 4 (Investor Syndication)
    syndication_evaluation = {"status": "MET", "explanation": "", "color": "🟢"}

    lead_investor_mentioned = "lead investor" in detailed_analysis.lower()

    if lead_investor_mentioned:
        syndication_evaluation = {"status": "MET", "explanation": "Lead investor identified in syndicate", "color": "🟢"}
    else:
        syndication_evaluation = {"status": "MET", "explanation": "No lead investor identified - not a rejection criterion per Kanoo Ventures policy", "color": "🟢"}

    criteria_evaluations["Investor Syndication"] = syndication_evaluation

    # STEP 8: Evaluate Criterion 5 (Fee Terms)
    fee_evaluation = {"status": "NOT MET", "explanation": "", "color": "🔴"}

    no_management_fees = "no management fee" in detailed_analysis.lower() or "no direct management fee" in detailed_analysis.lower()
    management_fees_present = "management fee" in detailed_analysis.lower() and not no_management_fees

    if no_management_fees:
        fee_evaluation = {"status": "MET", "explanation": "No direct management fees that would impact KV P&L", "color": "🟢"}
    elif management_fees_present:
        fee_evaluation = {"status": "NOT MET", "explanation": "Management fees present that would hit KV P&L", "color": "🔴"}
    else:
        fee_evaluation = {"status": "NOT MET", "explanation": "Fee information not mentioned - missing information counts as red", "color": "🔴"}

    criteria_evaluations["Fee Terms"] = fee_evaluation

    # STEP 9: Evaluate Criterion 6 (Investment Size)
    size_evaluation = {"status": "NOT MET", "explanation": "", "color": "🔴"}

    # Extract investment amounts
    import re
    amount_matches = re.findall(r'\$(\d+(?:\.\d+)?)\s*m', detailed_analysis.lower())
    investment_amount = 0

    if amount_matches:
        investment_amount = float(amount_matches[0])

    if investment_amount >= 7.9:
        size_evaluation = {"status": "MET", "explanation": f"Investment size ${investment_amount}m meets preferred threshold with strong preference noted", "color": "🟢"}
    elif investment_amount >= 5.0:
        size_evaluation = {"status": "MET", "explanation": f"Investment size ${investment_amount}m meets minimum threshold with note about preference for larger tickets", "color": "🟢"}
    elif investment_amount > 0 and investment_amount < 5.0:
        size_evaluation = {"status": "NOT MET", "explanation": f"Investment size ${investment_amount}m below $5m minimum - portfolio management concerns about too many small deals", "color": "🔴"}
    else:
        size_evaluation = {"status": "NOT MET", "explanation": "Investment size not specified", "color": "🔴"}

    criteria_evaluations["Investment Size"] = size_evaluation

    # STEP 10: Evaluate Criterion 7 (Process Timeline)
    timeline_evaluation = {"status": "NOT MET", "explanation": "", "color": "🔴"}

    # Extract timeline information
    timeline_weeks = 0
    week_matches = re.findall(r'(\d+)\s*week', detailed_analysis.lower())
    if week_matches:
        timeline_weeks = int(week_matches[0])

    is_kgi_coinvestment = kgi_found

    if is_kgi_coinvestment and timeline_weeks >= 3:
        timeline_evaluation = {"status": "MET", "explanation": f"KGI co-investment with {timeline_weeks} week timeline meets lighter diligence requirements", "color": "🟢"}
    elif timeline_weeks >= 8:
        timeline_evaluation = {"status": "MET", "explanation": f"Timeline of {timeline_weeks} weeks meets standard deal requirements", "color": "🟢"}
    elif timeline_weeks > 0:
        timeline_evaluation = {"status": "NOT MET", "explanation": f"Timeline of {timeline_weeks} weeks too short - risk of reduced diligence quality", "color": "🔴"}
    else:
        timeline_evaluation = {"status": "NOT MET", "explanation": "Timeline information absent", "color": "🔴"}

    criteria_evaluations["Process Timeline"] = timeline_evaluation

    # STEP 11: Evaluate Criterion 8 (Return Threshold)
    return_evaluation = {"status": "NOT MET", "explanation": "", "color": "🔴"}

    # Extract IRR information
    irr_matches = re.findall(r'irr.*?(\d+(?:\.\d+)?)\s*%', detailed_analysis.lower())
    irr_percentage = 0

    if irr_matches:
        irr_percentage = float(irr_matches[0])

    low_risk_mentioned = "low risk" in detailed_analysis.lower() or "low-risk" in detailed_analysis.lower()

    if irr_percentage >= 15:
        return_evaluation = {"status": "MET", "explanation": f"IRR of {irr_percentage}% meets 15% threshold", "color": "🟢"}
    elif irr_percentage > 0 and irr_percentage < 15 and low_risk_mentioned:
        return_evaluation = {"status": "MET", "explanation": f"IRR of {irr_percentage}% below 15% but justified as low-risk opportunity", "color": "🟢"}
    elif irr_percentage > 0 and irr_percentage < 15:
        return_evaluation = {"status": "NOT MET", "explanation": f"IRR of {irr_percentage}% below 15% without low-risk justification", "color": "🔴"}
    else:
        return_evaluation = {"status": "NOT MET", "explanation": "Return projections not provided", "color": "🔴"}

    criteria_evaluations["Return Threshold"] = return_evaluation

    # STEP 12: Evaluate Criterion 9 (Sector Focus)
    sector_evaluation = {"status": "NOT MET", "explanation": "", "color": "🔴"}

    target_sectors = ["healthcare", "education", "data economy", "energy transition", "industrials"]
    consumer_traditional = ["consumer", "traditional infrastructure"]

    sector_found = ""
    for sector in target_sectors:
        if sector in detailed_analysis.lower():
            sector_found = sector
            break

    consumer_found = any(sector in detailed_analysis.lower() for sector in consumer_traditional)

    if sector_found:
        sector_evaluation = {"status": "MET", "explanation": f"Company operates in {sector_found.title()} - target sector", "color": "🟢"}
    elif consumer_found:
        sector_evaluation = {"status": "NOT MET", "explanation": "Company in consumer or traditional infrastructure sectors", "color": "🔴"}
    else:
        # Check if meets other criteria for opportunistic consideration
        met_criteria_count = sum(1 for criteria in criteria_evaluations.values() if criteria["status"] == "MET")
        if met_criteria_count >= 6:  # Assuming most other criteria are met
            sector_evaluation = {"status": "MET", "explanation": "Opportunistic - meets other criteria and not in excluded sectors", "color": "🟢"}
        else:
            sector_evaluation = {"status": "NOT MET", "explanation": "Sector information not clear", "color": "🔴"}

    criteria_evaluations["Sector Focus"] = sector_evaluation

    # STEP 13: Generate comprehensive investment screening report
    current_date = datetime.now().strftime("%B %d, %Y")

    # Extract company name from analysis
    company_name = "Unknown Company"
    company_matches = re.search(r'company name[:\s]+([^\n\r\.]+)', detailed_analysis, re.IGNORECASE)
    if company_matches:
        company_name = company_matches.group(1).strip()

    # Count met vs not met criteria
    met_count = sum(1 for criteria in criteria_evaluations.values() if criteria["status"] == "MET")
    total_count = len(criteria_evaluations)

    # Generate overall recommendation
    if met_count >= 7:
        overall_recommendation = "RECOMMEND for further due diligence"
    elif met_count >= 5:
        overall_recommendation = "CONDITIONAL RECOMMEND - address key gaps"
    else:
        overall_recommendation = "DO NOT RECOMMEND - insufficient criteria met"

    report_prompt = f"""Generate a comprehensive investment screening report with the following information:

COMPANY: {company_name}
DATE: {current_date}
ANALYSIS: {detailed_analysis}
CRITERIA RESULTS: {json.dumps(criteria_evaluations, indent=2)}
MET CRITERIA: {met_count}/{total_count}
RECOMMENDATION: {overall_recommendation}

Format the report exactly as follows:

# INVESTMENT OPPORTUNITY SCREENING REPORT
**Date:** {current_date}

## {company_name}

### Executive Summary
[Provide 3-5 sentence overview of the opportunity including business model, investment size, and key highlights]

### Detailed Opportunity Summary
[Provide comprehensive business description, market opportunity, team background if available, and unique value proposition]

### Investment Criteria Evaluation

| Criterion | Evaluation |
|-----------|------------|
| 🟢/🔴 Geography/Structure | [Detailed explanation] |
| 🟢/🔴 Financial Milestones | [Detailed explanation] |
| 🟢/🔴 Asset Class Exclusion | [Detailed explanation] |
| 🟢/🔴 Investor Syndication | [Detailed explanation] |
| 🟢/🔴 Fee Terms | [Detailed explanation] |
| 🟢/🔴 Investment Size | [Detailed explanation] |
| 🟢/🔴 Process Timeline | [Detailed explanation] |
| 🟢/🔴 Return Threshold | [Detailed explanation] |
| 🟢/🔴 Sector Focus | [Detailed explanation] |

### Overall Recommendation
{overall_recommendation}

**Criteria Met:** {met_count} out of {total_count}

### Key Risks and Considerations
[List any applicable risks and considerations]

---
*Report generated by Kanoo Ventures Investment Screening System*"""

    final_report = await paradigm_client.chat_completion(report_prompt)

    # STEP 14: Return the formatted investment screening report
    return final_report