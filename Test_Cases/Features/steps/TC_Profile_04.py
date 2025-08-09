# ***************************************************************************************************
# File        : TC_Profile_04.py
# Description : Step definitions for user profile & logout scenario on ZARA site with AI healing support.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-07-18
# ***************************************************************************************************

from behave import given, when, then
from Common_Functions.CommonFunctions import (
    click,
    type_text,
    verify_element,
    add_screenshot_to_report,
    smart_find_element,
    verify_error_method,
)

@given('the user is on the ZARA homepage')
def step_user_on_homepage(context):
    tc_id = context.config.userdata["testcase"]
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Reject_Optional_Cookies', test_case_id=tc_id)
    click(context.driver, 'LOG_IN', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Logon_Button', test_case_id=tc_id)

@when('the user enters a valid email and password')
def step_user_enters_valid_credentials(context):
    tc_id = context.config.userdata["testcase"]
    email = context.testdata.get("Email", "default@example.com")
    pwd = context.testdata.get("Passcode", "defaultPass")
    type_text(context.driver, 'email_field', email, test_case_id=tc_id)
    type_text(context.driver, 'Password_field', pwd, test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Logon_Submit', test_case_id=tc_id)

@when('the user clicks on the profile icon')
def step_user_clicks_profile_icon(context):
    tc_id = context.config.userdata["testcase"]
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'UserProfileLink', test_case_id=tc_id)
    elem, loc, healed = smart_find_element(context.driver, 'header_user_account', test_case_id=tc_id)
    actual_text = elem.text
    expected_text = "Balaji"
    verify_error_method(actual_text, expected_text)
    verify_element(context.driver, 'header_user_account', test_case_id=tc_id)

@then('the user should be able to logout from ZARA')
def step_user_logs_out(context):
    tc_id = context.config.userdata["testcase"]
    click(context.driver, 'Logout_Button', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
