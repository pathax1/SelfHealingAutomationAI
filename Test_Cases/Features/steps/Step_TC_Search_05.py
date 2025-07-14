# ***************************************************************************************************
# File        : TC_Search_Product.py
# Description : Step definitions for IKEA Search Functionality on ZARA site with AI healing support.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-06-03
# ***************************************************************************************************

from behave import given, when, then
from selenium.webdriver.support.wait import WebDriverWait

from Common_Functions.CommonFunctions import click, type_text, verify_element, add_screenshot_to_report, \
    smart_find_element, verify_error_method
from selenium.webdriver.common.keys import Keys

@given('the user is on the ZARA HP')
def step_user_on_zara_hp(context):
    tc_id = context.config.userdata.get("testcase")  # Get the current test case ID
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Reject_Optional_Cookies', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)

@when('the user searches for "{search_query}"')
def step_user_search_product(context, search_query):
    tc_id = context.config.userdata.get("testcase")
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Search', test_case_id=tc_id)
    isearchproduct = context.testdata.get("ProductName", "0962/428/898")
    type_text(context.driver, 'search_home_input', isearchproduct, test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    # Fetch the element using smart healing so we can press ENTER
    elem, loc, healed = smart_find_element(context.driver, 'search_home_input', test_case_id=tc_id)
    # Press Enter
    elem.send_keys(Keys.ENTER)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)

@then('the results should include item related to "{product_code}"')
def step_verify_search_results(context, product_code):
    tc_id = context.config.userdata.get("testcase")
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Product_Info_Name', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    elem, loc, healed = smart_find_element(context.driver, 'product_name', test_case_id=tc_id)
    actual_text = elem.text
    print(actual_text)
    expected_text = "PRINTED HOODIE"
    verify_error_method(actual_text, expected_text)
