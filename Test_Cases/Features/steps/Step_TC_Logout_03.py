# ***************************************************************************************************
# File        : Step_TC_Logout_03.py
# Description : Step definitions for logout simulation scenario on ZARA site with AI healing support.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-07-18
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


@given('the user is logged into ZARA and has rejected cookies')
def step_reject_and_login(context):
    tc_id = context.config.userdata["testcase"]
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Reject_Optional_Cookies', test_case_id=tc_id)
    click(context.driver, 'LOG_IN', test_case_id=tc_id)
    click(context.driver, 'Logon_Button', test_case_id=tc_id)
    email = context.testdata.get("Email", "default@example.com")
    pwd = context.testdata.get("Passcode", "defaultPass")
    type_text(context.driver, 'email_field', email, test_case_id=tc_id)
    type_text(context.driver, 'Password_field', pwd, test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'Logon_Submit', test_case_id=tc_id)

@when('the user clicks on their profile link')
def step_click_profile(context):
    tc_id = context.config.userdata["testcase"]
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
    click(context.driver, 'UserProfileLink', test_case_id=tc_id)

@when('the user clicks the logout button')
def step_click_logout(context):
    tc_id = context.config.userdata["testcase"]
    click(context.driver, 'Logout_Button', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)

@then('the user should be redirected to the ZARA homepage')
def step_verify_homepage(context):
    tc_id = context.config.userdata["testcase"]
    # You can check the presence of an element only visible on the homepage, or validate URL, etc.
    verify_element(context.driver, 'header_user_account', test_case_id=tc_id)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, tc_id)
