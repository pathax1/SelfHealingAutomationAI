# ***************************************************************************************************
# File        : Step_TC_Login_02.py
# Description : Step definitions for invalid login scenario on ZARA site with AI healing support.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-06-10
# ***************************************************************************************************

from behave import given, when, then
from Common_Functions.CommonFunctions import (
    click,
    type_text,
    verify_element,
    verify_error_method,
    add_screenshot_to_report,
    smart_find_element
)

@given('the user has launched the ZARA site and reject cookies')
def step_reject_cookies(context):
    tc_id = context.config.userdata["testcase"]
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Reject_Optional_Cookies', test_case_id=tc_id)

@when('the user clicks on the login links')
def step_click_login_link(context):
    tc_id = context.config.userdata["testcase"]
    click(context.driver, 'LOG_IN', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Logon_Button', test_case_id=tc_id)

@when('the user enters an invalid email and password')
def step_enter_invalid_credentials(context):
    tc_id = context.config.userdata["testcase"]
    email = context.testdata.get("Email", "default@example.com")
    pwd = context.testdata.get("Passcode", "defaultPass")
    type_text(context.driver, 'email_field', email, test_case_id=tc_id)
    type_text(context.driver, 'Password_field', pwd, test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)

@when('the user clicks on the login button')
def step_click_login_button(context):
    tc_id = context.config.userdata["testcase"]
    click(context.driver, 'Logon_Submit', test_case_id=tc_id)

from behave import then

@then('an error message should be displayed')
def step_verify_error_message(context):
    tc_id = context.config.userdata["testcase"]
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    elem, loc, healed = smart_find_element(context.driver, 'login_error_message', test_case_id=tc_id)
    actual_text = elem.text
    expected_text = "Sorry, something went wrong"
    print("Expected: {}".format(expected_text))
    print("Actual: {}".format(actual_text))
    # Use case-insensitive comparison
    if expected_text.strip().upper() not in actual_text.strip().upper():
        raise AssertionError(f"Expected '{expected_text}' (case-insensitive) in actual '{actual_text}'")
    verify_element(context.driver, 'login_error_message', test_case_id=tc_id)

