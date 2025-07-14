# ***************************************************************************************************
# File        : TC_Profile_04.py.py
# Description : Step definitions for logout simulation scenario on ZARA site with AI healing support.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-05-25
# ***************************************************************************************************

from behave import given, when, then
from Common_Functions.CommonFunctions import click, type_text, verify_element, add_screenshot_to_report, \
    smart_find_element, verify_error_method


@given('the user is on the ZARA homepage')
def user_on_login_page(context):
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
    click(context.driver, 'Reject_Cookies')
    click(context.driver, 'Login_btn1')
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
    click(context.driver, 'Login_btn2')

@when('the user enters an valid email and password')
def user_clicks_on_logout(context):
    email = context.testdata.get("Email", "default@example.com")
    pwd = context.testdata.get("Passcode", "defaultPass")
    type_text(context.driver, 'email_field', email)
    type_text(context.driver, 'Password_field', pwd)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
    click(context.driver, 'Login_btn3')

@when('the user clicks on the profile icon')
def user_redirect_homepage(context):
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
    click(context.driver, 'UserProfileLink')

    # Use your smart locator system to fetch the element
    elem, loc, healed = smart_find_element(context.driver, 'header_user_account')

    # Extract actual text
    actual_text = elem.text

    # Set expected text
    expected_text = "Balaji"

    # Call your centralized verification method
    verify_error_method(actual_text, expected_text)

    verify_element(context.driver, 'header_user_account')

@then('the user should be able to logout from ZARA')
def user_redirect_homepage(context):
    click(context.driver, 'Logout_Button')
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,
                             context.config.userdata["testcase"])
