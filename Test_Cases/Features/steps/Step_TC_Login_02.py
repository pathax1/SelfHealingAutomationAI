# ***************************************************************************************************
# File        : Step_TC_Login_02.py
# Description : Step definitions for invalid login scenario on ZARA site with AI healing support.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-05-25
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
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])
    click(context.driver, 'Reject_Cookies')

@when('the user clicks on the login links')
def step_click_login_link(context):
    click(context.driver, 'Login_btn1')
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])
    click(context.driver, 'Login_btn2')

@when('the user enters an invalid email and password')
def step_enter_invalid_credentials(context):
    email = context.testdata.get("Email", "default@example.com")
    pwd = context.testdata.get("Passcode", "defaultPass")
    type_text(context.driver, 'email_field', email)
    type_text(context.driver, 'Password_field', pwd)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])

@when('the user clicks on the login button')
def step_click_login_button(context):
    click(context.driver, 'Login_btn3')

@then('an error message should be displayed')
def step_verify_error_message(context):
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])
    elem, loc, healed = smart_find_element(context.driver, 'login_error_message')
    actual_text = elem.text
    expected_text = "SORRY, SOMETHING WENT WRONG"
    verify_error_method(actual_text, expected_text)
    verify_element(context.driver, 'login_error_message')
